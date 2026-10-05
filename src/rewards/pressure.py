from typing import Any, Dict
from .base import RewardComponent


class PressureReward(RewardComponent):
    """Recompensa basada en Presión (MaxPressure / PressLight, KDD 2019).
    
    Minimiza el desequilibrio entre vehículos entrantes y salientes
    normalizado por capacidad de carril. Tiene garantía matemática
    de estabilidad de red (Varaiya, 2013; Wei et al., 2019).
    
    Fórmula:
        P_i = Σ(x_in / C_in) - Σ(x_out / C_out)
        R = -|P_i| * pressure_weight
    
    Ventajas sobre penalización por colas:
    - Previene derrames (spillback) porque considera carriles de salida
    - Coordinación implícita: mi salida = entrada del vecino
    - Propiedad Markoviana satisfecha (snapshot, no historial)
    """
    
    def __init__(self, pressure_weight: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pressure_weight = pressure_weight
    
    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Calcula la recompensa basada en presión normalizada."""
        pressure = env_info.get('avg_pressure', 0.0)
        # Minimizar la presión absoluta promedio de la red
        return -abs(pressure) * self.pressure_weight
