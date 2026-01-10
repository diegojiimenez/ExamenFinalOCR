"""
Módulo core del sistema OCR
"""

from core.model_manager import ModelManager
from core.predictor import UniversalPredictor, predict_cli
from core.model_training import OCRModelTrainer, train_model

__all__ = [
    'ModelManager',
    'UniversalPredictor',
    'predict_cli',
    'OCRModelTrainer',
    'train_model'
]