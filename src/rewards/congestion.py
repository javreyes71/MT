"""Penalización por congestión — versión v2 (sin throughput, sin clip).

r_loc = −( w_q · mean_l(h_l / c_l) + w_w · mean_l(wait_l / wait_max) )

Cambios respecto a v1 (B5):
  - Eliminado el término throughput (no depende del agente, dominaba).
  - Eliminado el clip ±10 (se usa VecNormalize en el entrenamiento).
  - La normalización es por carril y por capacidad → escala independiente de la red.
"""

from typing import Any, Dict
from .base import RewardComponent


class CongestionPenalty(RewardComponent):
    """Penalización por colas y tiempos de espera, normalizada por carril."""

    def __init__(
        self,
        halt_weight: float = 0.5,
        wait_penalty_weight: float = 0.5,
        wait_threshold: float = 20.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.halt_weight = halt_weight
        self.wait_weight = wait_penalty_weight
        self.wait_threshold = wait_threshold

    def calculate(self, env_info: Dict[str, Any]) -> float:
        halt_counts = env_info.get("halt_counts", [])
        wait_times = env_info.get("wait_times", [])
        num_lanes = max(len(halt_counts), 1)

        # Media de vehículos detenidos por carril
        avg_halt = sum(halt_counts) / num_lanes

        # Media de exceso de espera por carril
        excess = [
            (wt - self.wait_threshold) / max(self.wait_threshold, 1.0)
            for wt in wait_times
            if wt > self.wait_threshold
        ]
        avg_excess = sum(excess) / num_lanes if excess else 0.0

        return -(avg_halt * self.halt_weight + avg_excess * self.wait_weight)
