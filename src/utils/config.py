import yaml
from typing import Dict, Any


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
