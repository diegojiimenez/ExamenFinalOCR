"""
Gestión de configuración y rutas del sistema
"""

import yaml
import os
from pathlib import Path


class Config:
    """
    Clase para manejar la configuración del sistema
    """
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Carga el archivo de configuración YAML"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"No se encontró {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key, default=None):
        """Obtiene un valor de configuración"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def get_path(self, path_key):
        """Obtiene una ruta y la convierte a Path"""
        path_str = self.get(f'paths.{path_key}')
        return Path(path_str) if path_str else None
    
    def get_all_dataset_paths(self):
        """Obtiene todas las rutas de datasets"""
        return {
            'dataset_ia': self.get_path('dataset_ia'),
            'dataset_completo': self.get_path('dataset_completo'),
            'dataset_completo2': self.get_path('dataset_completo2')
        }
    
    def get_referencia_paths(self):
        """Obtiene rutas de imágenes de referencia"""
        return {
            'mayusculas': self.get_path('mayusculas'),
            'minusculas': self.get_path('minusculas'),
            'numeros': self.get_path('numeros')
        }
    
    def verify_paths(self):
        """Verifica que todas las rutas existan"""
        paths_to_check = [
            'dataset', 'referencia', 'pruebas'
        ]
        
        missing_paths = []
        for path_key in paths_to_check:
            path = self.get_path(path_key)
            if path and not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            print("⚠️  Rutas no encontradas:")
            for p in missing_paths:
                print(f"   - {p}")
            return False
        
        print("✅ Todas las rutas de datos están correctas")
        return True


# Instancia global
config = Config()