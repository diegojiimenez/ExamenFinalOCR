"""
Módulo de reducción de ruido en imágenes
"""

import cv2
import numpy as np


class NoiseReducer:
    """
    Clase para aplicar técnicas de reducción de ruido.
    """
    
    def __init__(self, config):
        self.config = config
        self.method = config['preprocessing']['noise_reduction']['method']
        self.kernel_size = config['preprocessing']['noise_reduction']['kernel_size']
    
    def reduce(self, image):
        """
        Aplica reducción de ruido a una imagen.
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            numpy.ndarray: Imagen sin ruido
        """
        # TODO: Implementar métodos de reducción de ruido
        # - Filtro Gaussiano
        # - Filtro Bilateral
        # - Filtro de Mediana
        pass
