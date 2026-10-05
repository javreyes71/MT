from typing import Any, Dict
from .base import RewardComponent


class StabilityPenalty(RewardComponent):
    """Penalización por cambios de fase excesivos (Anti-Chattering).
    
    Evita que el agente cambie de fase en cada paso de decisión,
    lo cual causa pérdida de capacidad vial por tiempos amarillos
    frecuentes y confunde a los conductores.
    
    Cada cambio de fase recibe una penalización fija.
    """
    
    def __init__(self, change_penalty: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.change_penalty = change_penalty
    
    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Penaliza proporcionalmente al número de cambios de fase."""
        phase_changes = env_info.get('phase_changes', 0)
        num_tls = env_info.get('num_tls', 1)
        # Normalizar por número de semáforos
        normalized_changes = phase_changes / max(num_tls, 1)
        return -normalized_changes * self.change_penalty
