"""
Módulo core del sistema OCR
"""

from core.model_manager import ModelManager
from core.predictor import UniversalPredictor, predict_cli

__all__ = [
    'ModelManager',
    'UniversalPredictor',
    'predict_cli'
]