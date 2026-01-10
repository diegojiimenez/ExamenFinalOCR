"""
Preprocesamiento de imágenes para OCR
"""

import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocesa una imagen para OCR
    
    Args:
        image: Imagen en escala de grises
        
    Returns:
        Imagen preprocesada (binarizada)
    """
    # Si es BGR, convertir a gris
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Binarización adaptativa
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Operaciones morfológicas para limpiar
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def enhance_contrast(image):
    """Mejora el contraste de la imagen"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)