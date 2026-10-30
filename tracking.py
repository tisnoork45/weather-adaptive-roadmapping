"""Self-contained ByteTrack multi-object tracker implementation.

Implements the core ByteTrack algorithm (Zhang et al., 2022): a Kalman-filter
based motion model per track combined with a two-stage association strategy
that first matches high-confidence detections, then attempts to recover
tracks using low-confidence detections that would normally be discarded by a
naive confidence threshold. Fully self-contained (numpy/scipy only).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from models import Detection
from utils import get_logger

logger = get_logger(__name__)


def xyxy_to_cxcyah(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)


def cxcyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, a, h = state
    w = a * h
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)

    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter_area + 1e-6
    return inter_area / union


class KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking a single object in [cx,cy,a,h] space."""

    _next_id = 1

    def __init__(self, box: np.ndarray, class_id: int, class_name: str, score: float) -> None:
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.class_id, self.class_name, self.score = class_id, class_name, score

        self.ndim = 4
        self.state = np.zeros(2 * self.ndim, dtype=np.float32)
        self.state[: self.ndim] = xyxy_to_cxcyah(box)

        self.F = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self.F[i, self.ndim + i] = 1.0

        self.H = np.zeros((self.ndim, 2 * self.ndim), dtype=np.float32)
        for i in range(self.ndim):
            self.H[i, i] = 1.0

        self.P = np.eye(2 * self.ndim, dtype=np.float32) * 10.0
        self.Q = np.eye(2 * self.ndim, dtype=np.float32) * 0.5
        self.R = np.eye(self.ndim, dtype=np.float32) * 1.0

        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return cxcyah_to_xyxy(self.state[: self.ndim])

    def update(self, box: np.ndarray, score: float) -> None:
        z = xyxy_to_cxcyah(box)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(2 * self.ndim, dtype=np.float32) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.score = score

    def get_box(self) -> np.ndarray:
        return cxcyah_to_xyxy(self.state[: self.ndim])


class ByteTracker:
    """ByteTrack: associates both high- and low-confidence detections to tracks."""

    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30,
                 match_thresh: float = 0.8, frame_rate: int = 30, min_box_area: float = 10.0) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.tracked: List[KalmanBoxTracker] = []
        self.lost: List[KalmanBoxTracker] = []
        self.frame_id = 0

    def update(self, detections: List[Detection]) -> List[Detection]:
        self.frame_id += 1

        boxes = np.array([d.box for d in detections], dtype=np.float32) if detections \
            else np.empty((0, 4), dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32) if detections \
            else np.empty((0,), dtype=np.float32)

        if len(boxes) > 0:
            areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
            keep = areas >= self.min_box_area
            boxes, scores = boxes[keep], scores[keep]
            detections = [d for d, k in zip(detections, keep) if k]

        high_mask = scores >= self.track_thresh
        low_mask = ~high_mask

        high_dets = [detections[i] for i in range(len(detections)) if high_mask[i]]
        high_boxes = boxes[high_mask] if len(boxes) else boxes
        high_scores = scores[high_mask] if len(scores) else scores

        low_dets = [detections[i] for i in range(len(detections)) if low_mask[i]]
        low_boxes = boxes[low_mask] if len(boxes) else boxes
        low_scores = scores[low_mask] if len(scores) else scores

        for t in self.tracked:
            t.predict()
        for t in self.lost:
            t.predict()

        # Stage 1: high-confidence detections vs tracked AND lost tracks (ByteTrack's
        # "track pool"). Lost tracks used to be predicted and pruned but never
        # matched again, so a single missed detection ended a track for good and
        # the object came back under a new ID -- 23 of 42 IDs on the first 240
        # frames of road.mp4 were such fragments.
        pool = self.tracked + self.lost
        track_boxes = np.array([t.get_box() for t in pool], dtype=np.float32) \
            if pool else np.empty((0, 4), dtype=np.float32)
        matches, unmatched_tracks, unmatched_dets = self._associate(track_boxes, high_boxes, self.match_thresh)

        recovered = []
        for t_idx, d_idx in matches:
            pool[t_idx].update(high_boxes[d_idx], float(high_scores[d_idx]))
            high_dets[d_idx].track_id = pool[t_idx].id
            if pool[t_idx] in self.lost:
                recovered.append(pool[t_idx])
        self.lost = [t for t in self.lost if t not in recovered]
        self.tracked = self.tracked + recovered

        # Only tracks that were tracked last frame go on to stage 2 (low-confidence
        # detections extend a live track but never revive a lost one).
        remaining_tracks = [pool[i] for i in unmatched_tracks if pool[i] in self.tracked]

        # Stage 2: remaining tracks vs low-confidence detections
        remaining_boxes = np.array([t.get_box() for t in remaining_tracks], dtype=np.float32) \
            if remaining_tracks else np.empty((0, 4), dtype=np.float32)
        matches2, _, _ = self._associate(remaining_boxes, low_boxes, thresh=0.5)

        still_unmatched = list(remaining_tracks)
        for t_idx, d_idx in matches2:
            track_obj = remaining_tracks[t_idx]
            track_obj.update(low_boxes[d_idx], float(low_scores[d_idx]))
            low_dets[d_idx].track_id = track_obj.id
            if track_obj in still_unmatched:
                still_unmatched.remove(track_obj)

        for t in still_unmatched:
            if t in self.tracked:
                self.tracked.remove(t)
            if t not in self.lost:
                self.lost.append(t)

        self.tracked = [t for t in self.tracked if t.time_since_update == 0] + \
            [t for t in remaining_tracks if t.time_since_update == 0 and t not in self.tracked]
        seen_ids, deduped = set(), []
        for t in self.tracked:
            if t.id not in seen_ids:
                deduped.append(t)
                seen_ids.add(t.id)
        self.tracked = deduped

        for d_idx in unmatched_dets:
            new_track = KalmanBoxTracker(high_boxes[d_idx], high_dets[d_idx].class_id,
                                          high_dets[d_idx].class_name, float(high_scores[d_idx]))
            high_dets[d_idx].track_id = new_track.id
            self.tracked.append(new_track)

        self.lost = [t for t in self.lost if t.time_since_update <= self.max_time_lost]
        return high_dets + low_dets

    @staticmethod
    def _associate(track_boxes: np.ndarray, det_boxes: np.ndarray, thresh: float
                    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

        iou_matrix = iou_batch(track_boxes, det_boxes)
        cost_matrix = 1.0 - iou_matrix
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matches, unmatched_tracks, unmatched_dets = [], set(range(len(track_boxes))), set(range(len(det_boxes)))
        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= (1.0 - thresh):
                matches.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)
        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)
