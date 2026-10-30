# Metric-depth variant of Depth Anything V2, following the official
# metric_depth/depth_anything_v2/dpt.py (https://github.com/DepthAnything/Depth-Anything-V2,
# Apache-2.0). That file is identical to the relative dpt.py except that the
# head ends in a Sigmoid instead of ReLU and the output is multiplied by
# max_depth, giving depth in metres in (0, max_depth). Neither activation has
# parameters, so the metric checkpoints load into this subclass with the same
# state-dict keys.

import torch.nn as nn

from .dpt import DepthAnythingV2


class MetricDepthAnythingV2(DepthAnythingV2):
    def __init__(self, encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024],
                 use_bn=False, use_clstoken=False, max_depth=20.0):
        super().__init__(encoder=encoder, features=features, out_channels=out_channels,
                         use_bn=use_bn, use_clstoken=use_clstoken)
        self.max_depth = max_depth
        # relative head: [conv, ReLU, conv, ReLU, Identity] -> metric head: [conv, ReLU, conv, Sigmoid, Identity]
        self.depth_head.scratch.output_conv2[3] = nn.Sigmoid()

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w) * self.max_depth
        return depth.squeeze(1)
