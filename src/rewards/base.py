from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type

class RewardComponent(ABC):
    """Clase base abstracta para componentes de recompensa."""
    
    def __init__(self, weight: float = 1.0, enabled: bool = True, **kwargs: Any) -> None:
        self.weight = weight
        self.enabled = enabled
    
    @abstractmethod
    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Calcula la contribución global de este componente al reward."""
        pass

    def calculate_agent(self, agent_id: str, agent_info: Dict[str, Any], global_info: Dict[str, Any]) -> float:
        """Calcula la recompensa individual para un agente (fallback a la global si no se implementa)."""
        return self.calculate(global_info)
    
    def __call__(self, env_info: Dict[str, Any]) -> float:
        if not self.enabled:
            return 0.0
        return self.calculate(env_info) * self.weight

class RewardManager:
    """Gestor que combina múltiples componentes de recompensa."""
    
    def __init__(self) -> None:
        self.components: List[RewardComponent] = []
    
    def add(self, component: RewardComponent) -> 'RewardManager':
        """Añade un componente de recompensa al manager."""
        self.components.append(component)
        return self

    def calculate_per_agent(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """Calcula la recompensa de manera desagregada por cada agente."""
        agents_info = env_info.get("agents_info", {})
        rewards = {aid: 0.0 for aid in agents_info.keys()}
        for c in self.components:
            if c.enabled:
                for aid, ainfo in agents_info.items():
                    rewards[aid] += c.calculate_agent(aid, ainfo, env_info) * c.weight
        return rewards
    
    def calculate_total(self, env_info: Dict[str, Any]) -> float:
        """Calcula la recompensa total (media global de todos los semáforos)."""
        per_agent = self.calculate_per_agent(env_info)
        return sum(per_agent.values()) / max(len(per_agent), 1)
    
    def get_breakdown(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """Retorna el desglose por componente (útil para debugging)."""
        return {type(c).__name__: c(env_info) for c in self.components}

    @classmethod
    def from_config(cls, reward_config: Dict[str, Any]) -> 'RewardManager':
        """Construye el RewardManager desde la configuración YAML."""
        manager = cls()
        
        # Importación tardía para evitar ciclos
        from .congestion import CongestionPenalty
        from .emissions import CO2Penalty
        from .pressure import PressureReward
        from .stability import StabilityPenalty
        from .neighborhood import NeighborhoodPressureReward
        from .eco_delay import EcoDelayReward
        
        component_map: Dict[str, Type[RewardComponent]] = {
            'congestion': CongestionPenalty,
            'emissions': CO2Penalty,
            'pressure': PressureReward,
            'stability': StabilityPenalty,
            'neighborhood': NeighborhoodPressureReward,
            'eco_delay': EcoDelayReward
        }
        
        components_conf = reward_config.get('components', {})
        for name, config_kwargs in components_conf.items():
            if name in component_map:
                comp_class = component_map[name]
                manager.add(comp_class(**config_kwargs))
                
        return manager
