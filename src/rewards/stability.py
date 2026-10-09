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
    
    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        """Penalización individual: penaliza si este semáforo cambió de fase."""
        is_yellow = agent_info.get("is_yellow", False)
        return -self.change_penalty if is_yellow else 0.0

    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Fallback global: penaliza proporcionalmente al número de cambios."""
        agents_info = env_info.get("agents_info", {})
        if not agents_info:
            phase_changes = env_info.get('phase_changes', 0)
            num_tls = env_info.get('num_tls', 1)
            normalized_changes = phase_changes / max(num_tls, 1)
            return -normalized_changes * self.change_penalty
        total = sum(self.calculate_agent(aid, info, env_info) for aid, info in agents_info.items())
        return total / max(len(agents_info), 1)
