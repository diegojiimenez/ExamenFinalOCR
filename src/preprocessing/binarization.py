"""
Módulo de binarización de imágenes
"""

import cv2
import numpy as np


class Binarizer:
    """
    Clase para binarizar imágenes (convertir a blanco y negro).
    """
    
    def __init__(self, config):
        self.config = config
        self.method = config['preprocessing']['binarization']['method']
    
    def binarize(self, image):
        """
        Binariza una imagen usando el método configurado.
        
        Args:
            image (numpy.ndarray): Imagen en escala de grises
            
        Returns:
            numpy.ndarray: Imagen binarizada
        """
        # TODO: Implementar métodos de binarización
        # - Método de Otsu
        # - Binarización adaptativa
        # - Método de Sauvola
        pass
