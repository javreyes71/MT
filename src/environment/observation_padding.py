"""Observation Padding Wrapper (OPW) — versión v2.

Estandariza las observaciones de intersecciones heterogéneas a una
dimensión fija para Parameter Sharing con MaskablePPO.

Estructura de la observación por agente (§3.1 del roadmap):
  - halts[max_lanes]:        cola normalizada por capacidad [0,1]
  - in_occ[max_lanes]:       ocupación entrante [0,1]
  - avg_out_occ:             ocupación saliente media [0,1]
  - phase_norm:              fase actual / (P_max-1) [0,1]
  - can_act_flag:            𝟙[τ ≥ g_min] {0,1}
  - time_norm:               τ / g_max [0,1]
  - local_pressure:          presión local normalizada [-1,1]
  - neighbor_pressure_avg:   presión media vecinos [0,1]
  - neighbor_queue_avg:      cola media vecinos [0,1]
  - type_onehot[n_groups]:   one-hot del grupo de intersección
  - lane_mask[max_lanes]:    máscara de carriles válidos {0,1}
  - phase_mask[max_phases]:  máscara de fases válidas {0,1}
"""

import numpy as np
from typing import List


class ObservationPaddingWrapper:
    """OPW v2: padding + máscara para intersecciones heterogéneas."""

    def __init__(
        self,
        max_lanes: int,
        max_neighbors: int,
        max_phases: int = 9,
        n_groups: int = 5,
    ) -> None:
        self.max_lanes = max_lanes
        self.max_neighbors = max_neighbors
        self.max_phases = max_phases
        self.n_groups = n_groups

        # Cálculo de la dimensión total
        # halts(max_lanes) + in_occ(max_lanes) + avg_out_occ(1)
        # + phase_norm(1) + can_act(1) + time_norm(1) + pressure(1)
        # + neighbor_pressure(1) + neighbor_queue(1)
        # + type_onehot(n_groups)
        # + lane_mask(max_lanes) + phase_mask(max_phases)
        self.padded_obs_dim = (
            max_lanes * 2      # halts + in_occ
            + 1                # avg_out_occ
            + 1                # phase_norm
            + 1                # can_act_flag
            + 1                # time_norm
            + 1                # local_pressure
            + 1                # neighbor_pressure_avg
            + 1                # neighbor_queue_avg
            + n_groups         # type_onehot
            + max_lanes        # lane_mask
            + max_phases       # phase_mask
        )

    def build_observation(
        self,
        halts: List[float],
        in_occ: List[float],
        avg_out_occ: float,
        phase_norm: float,
        can_act: bool,
        time_norm: float,
        local_pressure: float,
        neighbor_pressure_avg: float,
        neighbor_queue_avg: float,
        group_idx: int,
        actual_lanes: int,
        actual_phases: int,
    ) -> np.ndarray:
        """Construye la observación padded completa.

        Todos los inputs deben estar ya normalizados a [0,1] o [-1,1].
        """
        obs = np.zeros(self.padded_obs_dim, dtype=np.float32)
        offset = 0

        # halts (padded a max_lanes)
        n = min(len(halts), self.max_lanes)
        obs[offset: offset + n] = halts[:n]
        offset += self.max_lanes

        # in_occ (padded a max_lanes)
        n = min(len(in_occ), self.max_lanes)
        obs[offset: offset + n] = in_occ[:n]
        offset += self.max_lanes

        # avg_out_occ
        obs[offset] = np.clip(avg_out_occ, 0.0, 1.0)
        offset += 1

        # phase_norm
        obs[offset] = np.clip(phase_norm, 0.0, 1.0)
        offset += 1

        # can_act flag
        obs[offset] = 1.0 if can_act else 0.0
        offset += 1

        # time_norm
        obs[offset] = np.clip(time_norm, 0.0, 1.0)
        offset += 1

        # local_pressure (clipped a [-1,1])
        obs[offset] = np.clip(local_pressure, -1.0, 1.0)
        offset += 1

        # neighbor_pressure_avg
        obs[offset] = np.clip(neighbor_pressure_avg, 0.0, 1.0)
        offset += 1

        # neighbor_queue_avg
        obs[offset] = np.clip(neighbor_queue_avg, 0.0, 1.0)
        offset += 1

        # type_onehot
        if 0 <= group_idx < self.n_groups:
            obs[offset + group_idx] = 1.0
        offset += self.n_groups

        # lane_mask
        obs[offset: offset + actual_lanes] = 1.0
        offset += self.max_lanes

        # phase_mask
        obs[offset: offset + actual_phases] = 1.0
        offset += self.max_phases

        return obs

    def get_phase_mask(self, obs: np.ndarray) -> np.ndarray:
        """Extrae la máscara de fases válidas de una observación."""
        start = self.padded_obs_dim - self.max_phases
        return obs[start:].copy()

    # Backward compatibility
    def pad_observation(
        self, raw_obs: np.ndarray, actual_lanes: int, actual_neighbors: int
    ) -> np.ndarray:
        """Fallback: copia datos crudos y añade máscara de carriles."""
        padded = np.zeros(self.padded_obs_dim, dtype=np.float32)
        obs_len = min(len(raw_obs), self.padded_obs_dim - self.max_lanes - self.max_phases)
        padded[:obs_len] = raw_obs[:obs_len]
        # lane mask
        mask_start = self.padded_obs_dim - self.max_lanes - self.max_phases
        padded[mask_start: mask_start + actual_lanes] = 1.0
        return padded

    def normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """No-op: la observación v2 ya se construye normalizada."""
        return obs
