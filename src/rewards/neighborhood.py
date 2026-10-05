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

    def calculate(self, env_info: Dict[str, Any]) -> float:
        local_pressure = abs(env_info.get("avg_pressure", 0.0))
        neighbor_pressures = env_info.get("neighbor_pressures", [])

        r_local = -local_pressure

        if neighbor_pressures:
            avg_neighbor = sum(abs(p) for p in neighbor_pressures) / len(
                neighbor_pressures
            )
            r_neighbor = -avg_neighbor
        else:
            r_neighbor = r_local

        return (1 - self.beta) * r_local + self.beta * r_neighbor
