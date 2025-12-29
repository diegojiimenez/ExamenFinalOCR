"""
Módulo de gestión de configuración
"""

import yaml


def load_config(config_path="config.yaml"):
    """
    Carga la configuración desde un archivo YAML.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        dict: Configuración cargada
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
