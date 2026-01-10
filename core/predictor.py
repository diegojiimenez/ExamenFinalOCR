"""
Motor de predicción universal - Detecta y reconoce texto automáticamente
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image
from utils.segmentation import segment_characters
from core.model_manager import ModelManager
from utils.config import config


class UniversalPredictor:
    """
    Predictor universal que maneja letras, palabras y frases
    """
    
    def __init__(self, model_path=None):
        # Usar configuración si no se especifica ruta
        if model_path is None:
            model_path = config.get('model.path', 'models/best_model.h5')
        
        self.model_manager = ModelManager(model_path)
        self.model = self.model_manager.load_model()
        self.char_map = self._build_char_map()
    
    def _build_char_map(self):
        """Mapa de índices a caracteres desde configuración"""
        char_map_str = config.get('model.char_map', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return {i: char for i, char in enumerate(char_map_str)}
    
    def predict(self, image_path, debug=False):
        """
        Predicción universal con detección automática
        
        Args:
            image_path (str): Ruta de la imagen
            debug (bool): Mostrar información de debug
            
        Returns:
            tuple: (texto_reconocido, info_debug)
        """
        # 1. CARGAR Y PREPROCESAR
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo cargar: {image_path}")
        
        processed = preprocess_image(image)
        
        # 2. SEGMENTAR CARACTERES
        char_boxes = segment_characters(processed, debug=debug)
        
        if not char_boxes:
            if debug:
                print("⚠️  No se detectaron caracteres")
            return "", {'text': '', 'type': 'VACÍO', 'num_chars': 0, 'boxes': []}
        
        if debug:
            print(f"🔍 Caracteres detectados: {len(char_boxes)}")
        
        # 3. CLASIFICAR TIPO DE CONTENIDO
        content_type = self._detect_content_type(len(char_boxes))
        
        if debug:
            print(f"📋 Tipo detectado: {content_type}")
        
        # 4. RECONOCER CADA CARÁCTER
        recognized_chars = []
        for i, (x, y, w, h) in enumerate(char_boxes):
            char_img = processed[y:y+h, x:x+w]
            
            # Normalizar a 28x28
            char_resized = cv2.resize(char_img, (28, 28))
            char_normalized = char_resized.astype('float32') / 255.0
            char_input = char_normalized.reshape(1, 28, 28, 1)
            
            # Predecir
            prediction = self.model.predict(char_input, verbose=0)
            char_idx = np.argmax(prediction)
            confidence = prediction[0][char_idx]
            
            char = self.char_map.get(char_idx, '?')
            recognized_chars.append(char)
            
            if debug:
                print(f"  Char {i+1}: '{char}' (conf: {confidence:.2%})")
        
        # 5. ENSAMBLAR TEXTO FINAL
        final_text = ''.join(recognized_chars)
        
        info = {
            'text': final_text,
            'type': content_type,
            'num_chars': len(char_boxes),
            'boxes': char_boxes
        }
        
        return final_text, info
    
    def _detect_content_type(self, num_chars):
        """Detecta el tipo de contenido según número de caracteres"""
        if num_chars == 0:
            return "VACÍO"
        elif num_chars == 1:
            return "LETRA"
        elif 2 <= num_chars <= 6:
            return "PALABRA"
        else:
            return "FRASE"


def predict_cli(image_path):
    """Función auxiliar para línea de comandos"""
    from pathlib import Path
    
    print(f"\n{'='*60}")
    print(f"🔍 Procesando: {Path(image_path).name}")
    print('='*60)
    
    predictor = UniversalPredictor()
    text, info = predictor.predict(image_path, debug=True)
    
    print(f"\n{'='*60}")
    print(f"✅ RESULTADO: '{text}'")
    print(f"📊 Tipo: {info['type']} | Caracteres: {info['num_chars']}")
    print('='*60 + "\n")
    
    return text