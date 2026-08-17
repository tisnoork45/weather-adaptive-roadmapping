"""NEW test functions to APPEND to the existing tests.py (do not replace the
file -- add these functions alongside the existing ones; no existing test
is modified).
"""

from __future__ import annotations

from dataclasses import dataclass

from relative_velocity import RelativeVelocityEstimator
from risk_engine import RiskEngine
from ttc_calculator import compute_ttc, compute_ttc_for_objects


@dataclass
class _FakeBEVObject:
    track_id: int
    x_m: float
    z_m: float
    class_name: str = "car"
    confidence: float = 0.9


# =============================================================================
# relative_velocity.py
# =============================================================================


def test_relative_velocity_recovers_known_closing_speed() -> None:
    estimator = RelativeVelocityEstimator(history_length=5, max_gap_seconds=2.0, ema_alpha=1.0)
    t, z = 0.0, 50.0
    closing_speed = 5.0
    for dt in (0.30, 0.45, 0.28, 0.50, 0.33):  # variable real dt, not fixed frame rate
        t += dt
        z -= closing_speed * dt
        result = estimator.update([_FakeBEVObject(1, 2.0, z)], timestamp=t)
    vel = result[1]
    assert vel.valid
    assert abs(vel.vx_m_s - 0.0) < 0.05
    assert abs(vel.vz_m_s - (-closing_speed)) < 0.05


def test_relative_velocity_large_gap_is_invalid() -> None:
    estimator = RelativeVelocityEstimator(history_length=5, max_gap_seconds=2.0)
    estimator.update([_FakeBEVObject(1, 2.0, 50.0)], timestamp=0.0)
    estimator.update([_FakeBEVObject(1, 2.0, 45.0)], timestamp=0.5)
    result = estimator.update([_FakeBEVObject(1, 2.0, 44.0)], timestamp=5.5)  # large stall
    assert result[1].valid is False


def test_relative_velocity_prunes_lost_tracks() -> None:
    estimator = RelativeVelocityEstimator()
    estimator.update([_FakeBEVObject(1, 0, 0), _FakeBEVObject(2, 0, 0)], timestamp=0.0)
    assert set(estimator._history.keys()) == {1, 2}
    estimator.update([_FakeBEVObject(1, 0, 0)], timestamp=0.1)
    assert set(estimator._history.keys()) == {1}


def test_relative_velocity_needs_two_points() -> None:
    estimator = RelativeVelocityEstimator()
    result = estimator.update([_FakeBEVObject(1, 0, 10)], timestamp=0.0)
    assert result[1].valid is False


# =============================================================================
# ttc_calculator.py
# =============================================================================


def test_ttc_basic_math() -> None:
    assert compute_ttc(distance_m=50.0, closing_speed_m_s=5.0) == 10.0
    assert abs(compute_ttc(distance_m=3.0, closing_speed_m_s=6.0) - 0.5) < 1e-9


def test_ttc_none_when_not_closing() -> None:
    assert compute_ttc(distance_m=50.0, closing_speed_m_s=-2.0) is None
    assert compute_ttc(distance_m=50.0, closing_speed_m_s=0.01) is None


def test_ttc_capped_at_max() -> None:
    assert compute_ttc(distance_m=1000.0, closing_speed_m_s=0.2, max_ttc_s=60.0) == 60.0


def test_ttc_for_objects_skips_invalid_or_missing_velocity() -> None:
    from relative_velocity import RelativeVelocity

    objects = [
        _FakeBEVObject(track_id=1, x_m=0.0, z_m=20.0),
        _FakeBEVObject(track_id=2, x_m=0.0, z_m=30.0),  # no velocity entry
        _FakeBEVObject(track_id=3, x_m=0.0, z_m=15.0),  # velocity present but invalid
    ]
    velocities = {
        1: RelativeVelocity(track_id=1, vx_m_s=0.0, vz_m_s=-8.0, speed_m_s=8.0, valid=True),
        3: RelativeVelocity(track_id=3, vx_m_s=0.0, vz_m_s=-8.0, speed_m_s=8.0, valid=False),
    }
    results = compute_ttc_for_objects(objects, velocities)
    assert len(results) == 1
    assert results[0].track_id == 1
    assert abs(results[0].ttc_seconds - 2.5) < 1e-9


# =============================================================================
# risk_engine.py
# =============================================================================

_THRESHOLDS = {"CRITICAL": 1.5, "WARNING": 3.0, "CAUTION": 5.0}
_WEATHER_MULT = {"clear": 1.0, "fog": 1.6, "rain": 1.3, "night": 1.25, "snow": 1.5, "overcast": 1.05}


def _make_engine() -> RiskEngine:
    return RiskEngine(_THRESHOLDS, _WEATHER_MULT, min_safe_distance_m=5.0)


def test_risk_engine_ttc_boundaries() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    cases = [(1.0, "CRITICAL"), (2.0, "WARNING"), (4.0, "CAUTION"), (10.0, "SAFE")]
    for ttc, expected in cases:
        r = TTCResult(track_id=1, class_name="car", distance_m=20.0, closing_speed_m_s=5.0, ttc_seconds=ttc)
        result = engine.assess("clear", [r])
        assert result.per_object[0].risk_level == expected


def test_risk_engine_weather_adaptive_thresholds_genuinely_change_risk() -> None:
    """The core weather-adaptive requirement: identical TTC must classify
    differently depending on weather, not just carry a different number
    somewhere unused."""
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r = TTCResult(track_id=1, class_name="car", distance_m=20.0, closing_speed_m_s=5.0, ttc_seconds=2.0)
    clear_level = engine.assess("clear", [r]).per_object[0].risk_level
    fog_level = engine.assess("fog", [r]).per_object[0].risk_level
    assert clear_level == "WARNING"
    assert fog_level == "CRITICAL"
    assert clear_level != fog_level


def test_risk_engine_distance_floor_catches_stationary_close_object() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r = TTCResult(track_id=2, class_name="car", distance_m=2.0, closing_speed_m_s=-1.0, ttc_seconds=None)
    result = engine.assess("clear", [r])
    assert result.per_object[0].risk_level == "WARNING"


def test_risk_engine_overall_is_worst_object() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r_safe = TTCResult(track_id=3, class_name="person", distance_m=40.0, closing_speed_m_s=1.0, ttc_seconds=40.0)
    r_critical = TTCResult(track_id=4, class_name="car", distance_m=5.0, closing_speed_m_s=5.0, ttc_seconds=1.0)
    result = engine.assess("clear", [r_safe, r_critical])
    assert result.overall_risk_level == "CRITICAL"


def test_risk_engine_no_objects_is_safe() -> None:
    engine = _make_engine()
    result = engine.assess("clear", [])
    assert result.overall_risk_level == "SAFE"
    assert result.per_object == []


def test_risk_engine_unrecognized_weather_defaults_to_no_adjustment() -> None:
    from ttc_calculator import TTCResult

    engine = _make_engine()
    r = TTCResult(track_id=1, class_name="car", distance_m=20.0, closing_speed_m_s=5.0, ttc_seconds=2.0)
    result = engine.assess("some_unknown_weather_label", [r])
    assert result.per_object[0].risk_level == "WARNING"  # same as clear (multiplier defaults to 1.0)
