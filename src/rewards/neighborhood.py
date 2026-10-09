"""Recompensa con alineamiento de vecindario — versión v2 (CityLight).

r_i = (1−β) · r_i_loc + β · mean_{j∈N(i)} r_j_loc

donde r_loc = −( w_q · mean_l(h_l / c_l) + w_p · |P_i| )

β ≈ 0.3 fomenta la coordinación regional sin destruir el crédito local.
"""

from typing import Any, Dict
from .base import RewardComponent


class NeighborhoodPressureReward(RewardComponent):
    """Recompensa local + alineamiento de vecindario (CityLight, 2024).

    Usa el promedio ponderado entre la presión local y la presión
    media del vecindario para fomentar coordinación implícita.
    """

    def __init__(
        self,
        local_weight: float = 1.0,
        neighbor_weight: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.beta = neighbor_weight / (local_weight + neighbor_weight)

    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        """Recompensa individual con blending de vecindario."""
        local_pressure = abs(agent_info.get("pressure", 0.0))
        r_local = -local_pressure

        # Obtener presiones de los vecinos desde agents_info
        agents_info = global_info.get("agents_info", {})
        neighbor_pressures = []
        for aid, info in agents_info.items():
            if aid != agent_id:
                neighbor_pressures.append(abs(info.get("pressure", 0.0)))

        if neighbor_pressures:
            avg_neighbor = sum(neighbor_pressures) / len(neighbor_pressures)
            r_neighbor = -avg_neighbor
        else:
            r_neighbor = r_local

        return (1 - self.beta) * r_local + self.beta * r_neighbor

    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Fallback global: promedio de recompensas por agente."""
        agents_info = env_info.get("agents_info", {})
        if not agents_info:
            local_pressure = abs(env_info.get("avg_pressure", 0.0))
            r_local = -local_pressure
            neighbor_pressures = env_info.get("neighbor_pressures", [])
            if neighbor_pressures:
                avg_neighbor = sum(abs(p) for p in neighbor_pressures) / len(neighbor_pressures)
                r_neighbor = -avg_neighbor
            else:
                r_neighbor = r_local
            return (1 - self.beta) * r_local + self.beta * r_neighbor

        total = sum(self.calculate_agent(aid, info, env_info) for aid, info in agents_info.items())
        return total / max(len(agents_info), 1)
