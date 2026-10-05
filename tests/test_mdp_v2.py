"""Tests unitarios F3: observación, recompensa y OPW v2.

Verifica:
  - Forma y rango de la observación OPW v2
  - Signo de la recompensa (siempre ≤ 0 para congestion/pressure)
  - Determinismo: misma entrada → misma salida
  - Máscara de fases válida
  - Reward sin throughput ni clip
"""

import pytest
import numpy as np
from src.environment.observation_padding import ObservationPaddingWrapper
from src.rewards.congestion import CongestionPenalty
from src.rewards.pressure import PressureReward
from src.rewards.neighborhood import NeighborhoodPressureReward
from src.rewards.base import RewardManager


# =====================================================================
# Tests de ObservationPaddingWrapper v2
# =====================================================================

class TestOPWv2:
    def setup_method(self):
        self.opw = ObservationPaddingWrapper(
            max_lanes=8, max_neighbors=0, max_phases=4, n_groups=3
        )

    def test_padded_dim_correct(self):
        # 8*2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3 + 8 + 4 = 38
        expected = 8 * 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 3 + 8 + 4
        assert self.opw.padded_obs_dim == expected

    def test_build_observation_shape(self):
        obs = self.opw.build_observation(
            halts=[0.1, 0.2, 0.3],
            in_occ=[0.5, 0.6, 0.7],
            avg_out_occ=0.4,
            phase_norm=0.5,
            can_act=True,
            time_norm=0.8,
            local_pressure=-0.3,
            neighbor_pressure_avg=0.1,
            neighbor_queue_avg=0.2,
            group_idx=1,
            actual_lanes=3,
            actual_phases=2,
        )
        assert obs.shape == (self.opw.padded_obs_dim,)
        assert obs.dtype == np.float32

    def test_build_observation_range(self):
        """Toda la observación debe estar en [-1, 1]."""
        obs = self.opw.build_observation(
            halts=[0.9, 0.8],
            in_occ=[0.5, 0.6],
            avg_out_occ=0.9,
            phase_norm=1.0,
            can_act=False,
            time_norm=1.0,
            local_pressure=-0.9,
            neighbor_pressure_avg=0.5,
            neighbor_queue_avg=0.3,
            group_idx=2,
            actual_lanes=2,
            actual_phases=3,
        )
        assert np.all(obs >= -1.0), f"Min value: {obs.min()}"
        assert np.all(obs <= 1.0), f"Max value: {obs.max()}"

    def test_padding_zeros_for_missing_lanes(self):
        """Los carriles más allá de actual_lanes deben ser 0."""
        obs = self.opw.build_observation(
            halts=[0.5, 0.6],
            in_occ=[0.3, 0.4],
            avg_out_occ=0.2,
            phase_norm=0.0,
            can_act=True,
            time_norm=0.1,
            local_pressure=0.0,
            neighbor_pressure_avg=0.0,
            neighbor_queue_avg=0.0,
            group_idx=0,
            actual_lanes=2,
            actual_phases=2,
        )
        # halts: posiciones 2..7 deben ser 0
        assert np.all(obs[2:8] == 0.0)
        # in_occ: posiciones 10..15 deben ser 0
        assert np.all(obs[10:16] == 0.0)

    def test_can_act_flag(self):
        obs_can = self.opw.build_observation(
            halts=[], in_occ=[], avg_out_occ=0.0, phase_norm=0.0,
            can_act=True, time_norm=0.0, local_pressure=0.0,
            neighbor_pressure_avg=0.0, neighbor_queue_avg=0.0,
            group_idx=0, actual_lanes=0, actual_phases=2,
        )
        obs_cant = self.opw.build_observation(
            halts=[], in_occ=[], avg_out_occ=0.0, phase_norm=0.0,
            can_act=False, time_norm=0.0, local_pressure=0.0,
            neighbor_pressure_avg=0.0, neighbor_queue_avg=0.0,
            group_idx=0, actual_lanes=0, actual_phases=2,
        )
        # can_act is at position: max_lanes*2 + 1(out) + 1(phase) = 18
        can_act_idx = 8 * 2 + 1 + 1
        assert obs_can[can_act_idx] == 1.0
        assert obs_cant[can_act_idx] == 0.0

    def test_type_onehot(self):
        obs = self.opw.build_observation(
            halts=[], in_occ=[], avg_out_occ=0.0, phase_norm=0.0,
            can_act=False, time_norm=0.0, local_pressure=0.0,
            neighbor_pressure_avg=0.0, neighbor_queue_avg=0.0,
            group_idx=2, actual_lanes=0, actual_phases=2,
        )
        # type_onehot starts after: max_lanes*2 + 7 scalars
        oh_start = 8 * 2 + 7
        assert obs[oh_start + 2] == 1.0
        assert obs[oh_start + 0] == 0.0
        assert obs[oh_start + 1] == 0.0

    def test_phase_mask(self):
        obs = self.opw.build_observation(
            halts=[], in_occ=[], avg_out_occ=0.0, phase_norm=0.0,
            can_act=False, time_norm=0.0, local_pressure=0.0,
            neighbor_pressure_avg=0.0, neighbor_queue_avg=0.0,
            group_idx=0, actual_lanes=3, actual_phases=2,
        )
        mask = self.opw.get_phase_mask(obs)
        assert mask.shape == (4,)
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[2] == 0.0
        assert mask[3] == 0.0

    def test_lane_mask(self):
        obs = self.opw.build_observation(
            halts=[], in_occ=[], avg_out_occ=0.0, phase_norm=0.0,
            can_act=False, time_norm=0.0, local_pressure=0.0,
            neighbor_pressure_avg=0.0, neighbor_queue_avg=0.0,
            group_idx=0, actual_lanes=3, actual_phases=2,
        )
        # lane_mask starts at padded_obs_dim - max_phases - max_lanes
        lm_start = self.opw.padded_obs_dim - 4 - 8
        assert obs[lm_start] == 1.0
        assert obs[lm_start + 2] == 1.0
        assert obs[lm_start + 3] == 0.0  # padded

    def test_determinism(self):
        """Misma entrada → misma salida."""
        kwargs = dict(
            halts=[0.1, 0.2], in_occ=[0.3, 0.4], avg_out_occ=0.5,
            phase_norm=0.5, can_act=True, time_norm=0.6,
            local_pressure=-0.1, neighbor_pressure_avg=0.2,
            neighbor_queue_avg=0.3, group_idx=1,
            actual_lanes=2, actual_phases=3,
        )
        obs1 = self.opw.build_observation(**kwargs)
        obs2 = self.opw.build_observation(**kwargs)
        np.testing.assert_array_equal(obs1, obs2)


# =====================================================================
# Tests de Recompensas v2
# =====================================================================

class TestCongestionV2:
    def test_no_congestion_zero_reward(self):
        r = CongestionPenalty(halt_weight=0.5, wait_penalty_weight=0.5)
        result = r.calculate({
            "halt_counts": [0, 0, 0],
            "wait_times": [0.0, 0.0, 0.0],
        })
        assert result == 0.0

    def test_negative_with_congestion(self):
        r = CongestionPenalty(halt_weight=0.5, wait_penalty_weight=0.5)
        result = r.calculate({
            "halt_counts": [5, 10, 3],
            "wait_times": [30.0, 40.0, 10.0],
        })
        assert result < 0.0

    def test_no_throughput_term(self):
        """El throughput no debe afectar la recompensa (B5)."""
        r = CongestionPenalty(halt_weight=0.5, wait_penalty_weight=0.5)
        info_no_tp = {"halt_counts": [5], "wait_times": [30.0]}
        info_with_tp = {"halt_counts": [5], "wait_times": [30.0], "throughput": 100}
        assert r.calculate(info_no_tp) == r.calculate(info_with_tp)

    def test_no_hard_clip(self):
        """No debe haber clip ±10 (B5)."""
        r = CongestionPenalty(halt_weight=1.0, wait_penalty_weight=1.0, wait_threshold=0.0)
        result = r.calculate({
            "halt_counts": [100] * 10,
            "wait_times": [200.0] * 10,
        })
        assert result < -10.0  # Antes se clipeaba a -10


class TestPressureReward:
    def test_zero_pressure_zero_reward(self):
        r = PressureReward(pressure_weight=1.0)
        result = r.calculate({"avg_pressure": 0.0})
        assert result == 0.0

    def test_negative_with_pressure(self):
        r = PressureReward(pressure_weight=1.0)
        result = r.calculate({"avg_pressure": 0.5})
        assert result < 0.0


class TestNeighborhoodV2:
    def test_blending(self):
        r = NeighborhoodPressureReward(local_weight=1.0, neighbor_weight=0.3)
        result = r.calculate({
            "avg_pressure": 0.5,
            "neighbor_pressures": [0.3, 0.4],
        })
        # Should be negative (penalizing pressure)
        assert result < 0.0

    def test_no_neighbors_uses_local(self):
        r = NeighborhoodPressureReward(local_weight=1.0, neighbor_weight=0.3)
        result = r.calculate({
            "avg_pressure": 0.5,
            "neighbor_pressures": [],
        })
        assert result < 0.0


class TestRewardManagerV2:
    def test_from_config_no_emergency(self):
        """from_config no debe fallar sin emergency."""
        config = {
            "components": {
                "congestion": {"enabled": True, "halt_weight": 0.5, "wait_penalty_weight": 0.5},
                "pressure": {"enabled": True, "pressure_weight": 1.0},
            }
        }
        manager = RewardManager.from_config(config)
        assert len(manager.components) == 2

    def test_disabled_component_returns_zero(self):
        config = {
            "components": {
                "congestion": {"enabled": False, "halt_weight": 0.5, "wait_penalty_weight": 0.5},
            }
        }
        manager = RewardManager.from_config(config)
        result = manager.calculate_total({
            "halt_counts": [10, 20],
            "wait_times": [50.0, 60.0],
        })
        assert result == 0.0
