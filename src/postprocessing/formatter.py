"""
Módulo de formateo de salida
"""

import os
from datetime import datetime


class OutputFormatter:
    """
    Clase para formatear y guardar resultados del OCR.
    """
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config['paths']['output']
    
    def save(self, text, image_path, segments=None):
        """
        Guarda el texto reconocido en un archivo.
        
        Args:
            text (str): Texto reconocido
            image_path (str): Ruta de la imagen original
            segments (dict): Segmentos detectados (opcional)
            
        Returns:
            str: Ruta del archivo de salida
        """
        # TODO: Implementar guardado de resultados
        # - Crear nombre de archivo
        # - Guardar en formato .txt
        # - Opcionalmente guardar metadata en JSON
        pass
