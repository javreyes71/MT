from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type

class RewardComponent(ABC):
    """Clase base abstracta para componentes de recompensa."""
    
    def __init__(self, weight: float = 1.0, enabled: bool = True, **kwargs: Any) -> None:
        self.weight = weight
        self.enabled = enabled
    
    @abstractmethod
    def calculate(self, env_info: Dict[str, Any]) -> float:
        """Calcula la contribución de este componente al reward."""
        pass
    
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
    
    def calculate_total(self, env_info: Dict[str, Any]) -> float:
        """Calcula la recompensa total sumando todos los componentes."""
        return sum(c(env_info) for c in self.components)
    
    def get_breakdown(self, env_info: Dict[str, Any]) -> Dict[str, float]:
        """Retorna el desglose por componente (útil para debugging/TensorBoard)."""
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
        
        component_map: Dict[str, Type[RewardComponent]] = {
            'congestion': CongestionPenalty,
            'emissions': CO2Penalty,
            'pressure': PressureReward,
            'stability': StabilityPenalty,
            'neighborhood': NeighborhoodPressureReward
        }
        
        components_conf = reward_config.get('components', {})
        for name, config_kwargs in components_conf.items():
            if name in component_map:
                comp_class = component_map[name]
                manager.add(comp_class(**config_kwargs))
                
        return manager
