"""
Módulo de carga de imágenes
"""

import cv2
import numpy as np
from PIL import Image


class ImageLoader:
    """
    Clase para cargar y validar imágenes de entrada.
    """
    
    def __init__(self, config):
        """
        Inicializa el cargador de imágenes.
        
        Args:
            config (dict): Configuración del sistema
        """
        self.config = config
        self.allowed_formats = config['input']['allowed_formats']
    
    def load(self, image_path):
        """
        Carga una imagen desde una ruta.
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            numpy.ndarray: Imagen cargada
            
        Raises:
            ValueError: Si el formato no es válido
        """
        # TODO: Implementar carga de imagen
        # Verificar formato
        # Cargar con OpenCV o PIL
        # Validar dimensiones
        pass
