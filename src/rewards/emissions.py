from typing import Any, Dict
from .base import RewardComponent

class CO2Penalty(RewardComponent):
    """Penaliza emisiones de CO2 para incentivar flujo eficiente."""
    
    def __init__(self, co2_weight: float = 0.001, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.co2_weight = co2_weight
        
    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        """Penalización individual por CO2 de este semáforo."""
        co2 = float(agent_info.get("co2", 0.0))
        capacity = float(agent_info.get("capacity", 1.0))
        # Normalizar por capacidad para que la escala sea comparable entre intersecciones
        return -(co2 / max(capacity, 1.0)) * self.co2_weight

    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Fallback global: promedio de penalizaciones por agente."""
        agents_info = env_info.get("agents_info", {})
        if not agents_info:
            total_co2 = float(env_info.get('total_co2', 0.0))
            return -total_co2 * self.co2_weight
        total = sum(self.calculate_agent(aid, info, env_info) for aid, info in agents_info.items())
        return total / max(len(agents_info), 1)
