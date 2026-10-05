from typing import Any, Dict
from .base import RewardComponent

class CO2Penalty(RewardComponent):
    """Penaliza emisiones de CO2 para incentivar flujo eficiente."""
    
    def __init__(self, co2_weight: float = 0.001, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.co2_weight = co2_weight
        
    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Calcula la penalización por emisiones de CO2."""
        total_co2 = float(env_info.get('total_co2', 0.0))
        return -total_co2 * self.co2_weight  # Multiplicamos por weight si lo usamos
