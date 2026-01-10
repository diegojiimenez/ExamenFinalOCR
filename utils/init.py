"""
Módulo de utilidades
"""

from utils.config import config
from utils.data_loader import data_loader
from utils.preprocessing import preprocess_image, enhance_contrast
from utils.segmentation import segment_characters, visualize_boxes
from utils.dataset import LABEL_MAP, preprocess_char_unified

# Intentar importar advanced
try:
    from utils.enhanced_preprocessing import advanced_char_preprocessing, intelligent_segmentation
    __all__ = [
        'config',
        'data_loader',
        'preprocess_image',
        'enhance_contrast',
        'segment_characters',
        'visualize_boxes',
        'LABEL_MAP',
        'preprocess_char_unified',
        'advanced_char_preprocessing',
        'intelligent_segmentation'
    ]
except ImportError:
    __all__ = [
        'config',
        'data_loader',
        'preprocess_image',
        'enhance_contrast',
        'segment_characters',
        'visualize_boxes',
        'LABEL_MAP',
        'preprocess_char_unified'
    ]