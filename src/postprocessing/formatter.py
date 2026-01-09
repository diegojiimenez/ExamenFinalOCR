"""
Módulo de formateo de salida
"""

import os
import json
from datetime import datetime
from pathlib import Path


class OutputFormatter:
    """
    Clase para formatear y guardar resultados del OCR.
    """
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['paths']['output'])
        self.text_dir = self.output_dir / 'text'
        
        # Crear directorios
        self.text_dir.mkdir(parents=True, exist_ok=True)
    
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
        input_path = Path(image_path)
        base_name = input_path.stem
        
        # Guardar texto plano
        text_output = self.text_dir / f"{base_name}.txt"
        with open(text_output, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Guardar metadata si hay segmentos
        if segments:
            json_output = self.text_dir / f"{base_name}_metadata.json"
            metadata = {
                'input_file': str(image_path),
                'processed_date': datetime.now().isoformat(),
                'text': text,
                'statistics': {
                    'lines': len(segments.get('lines', [])),
                    'words': len(segments.get('words', [])),
                    'characters': len(segments.get('characters', []))
                }
            }
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return str(text_output)
