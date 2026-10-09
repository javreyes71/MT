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
    
    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        """Penalización individual por presión de este semáforo."""
        pressure = agent_info.get("pressure", 0.0)
        return -abs(pressure) * self.pressure_weight

    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Fallback global: presión promedio de toda la red."""
        agents_info = env_info.get("agents_info", {})
        if not agents_info:
            pressure = env_info.get('avg_pressure', 0.0)
            return -abs(pressure) * self.pressure_weight
        total = sum(self.calculate_agent(aid, info, env_info) for aid, info in agents_info.items())
        return total / max(len(agents_info), 1)
