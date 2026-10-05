import yaml
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class SimulationConfig:
    net_file: str = "sumo/network.net.xml"
    config_file: str = "sumo/simulation.sumocfg"
    delta_time: float = 5.0
    min_phase_time: float = 10.0
    yellow_time: float = 3.0
    max_wait_time: float = 20.0
    duration: int = 3600

@dataclass
class TrainingConfig:
    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    total_timesteps: int = 500000
    eval_freq: int = 50000
    checkpoint_freq: int = 10000
    seed: int = 42

@dataclass
class RewardConfig:
    components: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MARLConfig:
    mode: str = "centralized"
    neighbor_observation: bool = True
    communication_radius: int = 2

class ProjectConfig:
    """Clase principal que contiene todas las sub-configuraciones del proyecto."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.simulation = SimulationConfig(**config_dict.get('simulation', {}))
        self.training = TrainingConfig(**config_dict.get('training', {}))
        self.reward = RewardConfig(**config_dict.get('reward', {}))
        self.marl = MARLConfig(**config_dict.get('marl', {}))
        self.emergency = config_dict.get('emergency', {})
        self.paths = config_dict.get('paths', {})

def load_config(path: str = 'config/default.yaml') -> Dict[str, Any]:
    """Carga la configuración desde un archivo YAML y mezcla controlled_tls si existe."""
    import os
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    tls_path = os.path.join(os.path.dirname(path), 'controlled_tls.yaml')
    if os.path.exists(tls_path):
        with open(tls_path, 'r', encoding='utf-8') as file:
            tls_config = yaml.safe_load(file)
            if tls_config:
                config = merge_configs(config, tls_config)
    return config

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Combina dos diccionarios de configuración recursivamente."""
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
