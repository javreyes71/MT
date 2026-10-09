"""Recompensa Eco-Delay: Equilibra congestión (Delay) y emisiones (Eco)."""
from typing import Any, Dict
from .base import RewardComponent

class EcoDelayReward(RewardComponent):
    def __init__(self, delay_weight: float = 0.5, eco_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay_weight = delay_weight
        self.eco_weight = eco_weight

    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        halts = agent_info.get("halts", 0.0)
        capacity = agent_info.get("capacity", 1.0)
        co2 = agent_info.get("co2", 0.0)

        # Normalizado por capacidad del semáforo
        normalized_delay = halts / capacity
        
        # Estimación: ~2000 mg/s por vehículo detenido. Normalizamos.
        normalized_eco = co2 / (capacity * 2000.0)

        return -(self.delay_weight * normalized_delay + self.eco_weight * normalized_eco)

    def calculate(self, env_info: Dict[str, Any]) -> float:
        # Fallback global por si se llama desde un entorno mono-agente
        agents_info = env_info.get("agents_info", {})
        if not agents_info:
            return 0.0
        total = sum(self.calculate_agent(aid, info, env_info) for aid, info in agents_info.items())
        return total / max(len(agents_info), 1)
