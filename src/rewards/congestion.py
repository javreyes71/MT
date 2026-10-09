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

    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        """Penalización individual por congestión en este semáforo."""
        halts = agent_info.get("halts", 0)
        wait = agent_info.get("wait", 0.0)
        capacity = agent_info.get("capacity", 1.0)

        # Cola normalizada por capacidad del semáforo
        norm_halt = halts / max(capacity, 1.0)

        # Exceso de espera normalizado
        norm_wait = max(0.0, wait - self.wait_threshold) / max(self.wait_threshold, 1.0)

        return -(norm_halt * self.halt_weight + norm_wait * self.wait_weight)

    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Global: usa agents_info si existe, sino fallback a halt_counts/wait_times legacy."""
        agents_info = env_info.get("agents_info", {})
        if agents_info:
            total = sum(self.calculate_agent(aid, info, env_info) for aid, info in agents_info.items())
            return total / max(len(agents_info), 1)

        # Fallback legacy (para tests y entornos mono-agente)
        halt_counts = env_info.get("halt_counts", [])
        wait_times = env_info.get("wait_times", [])
        num_lanes = max(len(halt_counts), 1)

        avg_halt = sum(halt_counts) / num_lanes

        excess = [
            (wt - self.wait_threshold) / max(self.wait_threshold, 1.0)
            for wt in wait_times
            if wt > self.wait_threshold
        ]
        avg_excess = sum(excess) / num_lanes if excess else 0.0

        return -(avg_halt * self.halt_weight + avg_excess * self.wait_weight)
