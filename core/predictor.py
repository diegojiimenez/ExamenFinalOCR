"""
Motor de predicción universal - ADAPTADO DEL PROYECTO ORIGINAL
Compatible con modelo entrenado 32x32, 62 clases
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.models import load_model
from pathlib import Path

from core.model_manager import ModelManager
from utils.config import config
from utils.dataset import LABEL_MAP, preprocess_char_unified


class UniversalPredictor:
    """
    Predictor que replica exactamente la lógica del proyecto original
    """
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.get('model.path', 'models/trained_model.h5')
        
        self.model_manager = ModelManager(model_path)
        self.model = self.model_manager.load_model()
        
        # Configuración del modelo
        self.input_size = 32  # Forzar a 32x32
        self.num_classes = 62
        
        print(f"📊 Modelo cargado:")
        print(f"   Input size: {self.input_size}x{self.input_size}")
        print(f"   Clases: {self.num_classes}")
    
    def predict(self, image_path, debug=False, visualize=False):
        """
        Predicción usando la lógica EXACTA del proyecto original
        """
        print(f"\n🔍 Procesando: {image_path}")
        
        # Verificar archivo
        if not Path(image_path).exists():
            raise ValueError(f"No se encuentra: {image_path}")
        
        # Cargar imagen en escala de grises
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError(f"No se pudo cargar: {image_path}")
        
        h, w = original.shape
        print(f"📐 Imagen: {w}x{h}")
        
        # Detectar tipo de escritura (del original)
        variance = np.var(original)
        is_handwritten = variance > 800
        print(f"📝 Tipo: {'Manuscrita' if is_handwritten else 'Digital'} (var: {variance:.1f})")
        
        # PREPROCESAMIENTO ADAPTATIVO (del original)
        if is_handwritten:
            # Manuscritas: preprocesamiento suave
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(original)
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 15, 8
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        else:
            # Digitales: binarización estándar
            _, binary = cv2.threshold(
                original, 0, 255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        
        # SEGMENTACIÓN (del original)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtros adaptativos
        char_boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = cw / ch if ch > 0 else 0
            
            if is_handwritten:
                # Filtros permisivos para manuscritas
                min_area, min_dimension = 20, 5
                max_width, max_height = w * 0.6, h * 0.9
                aspect_range = (0.05, 6.0)
            else:
                # Filtros estrictos para digitales
                min_area, min_dimension = 50, 8
                max_width, max_height = w * 0.3, h * 0.8
                aspect_range = (0.1, 3.0)
            
            if (area >= min_area and 
                cw >= min_dimension and ch >= min_dimension and
                cw <= max_width and ch <= max_height and
                aspect_range[0] <= aspect_ratio <= aspect_range[1]):
                char_boxes.append((x, y, cw, ch))
        
        num_chars = len(char_boxes)
        print(f"🔍 Caracteres detectados: {num_chars}")
        
        if not char_boxes:
            return "", {'text': '', 'type': 'VACÍO', 'num_chars': 0}
        
        # DETECCIÓN DE TIPO DE CONTENIDO
        if num_chars == 1:
            content_type = "LETRA"
        elif 2 <= num_chars <= 6:
            content_type = "PALABRA"
        else:
            content_type = "FRASE"
        
        print(f"📋 Tipo: {content_type}")
        
        # ORDENAMIENTO INTELIGENTE (del original)
        char_boxes = self._smart_sort(char_boxes, content_type)
        
        # RECONOCIMIENTO
        recognized_chars = []
        confidences = []
        
        for i, (x, y, cw, ch) in enumerate(char_boxes):
            # Extraer carácter con margen
            margin = max(2, min(cw, ch) // 8)
            y_start = max(0, y - margin)
            y_end = min(h, y + ch + margin)
            x_start = max(0, x - margin)
            x_end = min(w, x + cw + margin)
            
            char_img = original[y_start:y_end, x_start:x_end]
            
            # Preprocesamiento del carácter
            char_processed = preprocess_char_unified(
                char_img, 
                debug=(debug and i < 2)
            )
            
            # Preparar para predicción: (1, 32, 32, 1)
            char_input = char_processed.reshape(1, 32, 32, 1)
            
            # Predecir
            prediction = self.model.predict(char_input, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Preprocesamiento alternativo si confianza baja
            if confidence < 0.3:
                char_alt = cv2.resize(char_img, (32, 32))
                if np.mean(char_alt) > 127:
                    char_alt = 255 - char_alt
                char_alt = char_alt / 255.0
                char_alt = np.where(char_alt > 0.15, 1.0, 0.0)
                
                char_alt_input = char_alt.reshape(1, 32, 32, 1)
                prediction_alt = self.model.predict(char_alt_input, verbose=0)
                confidence_alt = np.max(prediction_alt)
                
                if confidence_alt > confidence:
                    prediction = prediction_alt
                    predicted_class = np.argmax(prediction)
                    confidence = confidence_alt
                    if debug:
                        print(f"   🔄 Preprocesamiento alternativo: char {i+1}")
            
            # Decodificar
            predicted_char = LABEL_MAP.get(predicted_class, '?')
            recognized_chars.append(predicted_char)
            confidences.append(confidence)
            
            print(f"   Char {i+1}: '{predicted_char}' ({confidence:.1%})")
        
        # CONSTRUCCIÓN DE TEXTO (del original)
        if content_type == "LETRA":
            phrase = "".join(recognized_chars)
        elif content_type == "PALABRA":
            phrase = "".join(recognized_chars)
        else:  # FRASE
            phrase = self._build_phrase_with_spaces(
                recognized_chars, char_boxes
            )
        
        print(f"\n📝 Resultado: '{phrase}'")
        
        info = {
            'text': phrase,
            'type': content_type,
            'num_chars': num_chars,
            'boxes': char_boxes,
            'confidences': confidences,
            'recognized_chars': recognized_chars
        }
        
        # Visualización si se solicita
        if visualize:
            fig = self._create_visualization(
                original, char_boxes, recognized_chars, 
                confidences, phrase, content_type
            )
            return phrase, info, fig
        
        return phrase, info
    
    def _smart_sort(self, boxes, content_type):
        """Ordenamiento inteligente de caracteres"""
        if not boxes or content_type == "LETRA":
            return boxes
        
        # Agrupar por líneas
        avg_height = np.mean([h for _, _, _, h in boxes])
        line_tolerance = avg_height * 0.5
        
        lines = []
        for box in boxes:
            x, y, w, h = box
            y_center = y + h // 2
            
            placed = False
            for line in lines:
                line_y_avg = np.mean([b[1] + b[3]//2 for b in line])
                if abs(y_center - line_y_avg) <= line_tolerance:
                    line.append(box)
                    placed = True
                    break
            
            if not placed:
                lines.append([box])
        
        # Ordenar líneas por Y
        lines.sort(key=lambda line: np.mean([b[1] for b in line]))
        
        # Ordenar caracteres por X dentro de cada línea
        for line in lines:
            line.sort(key=lambda b: b[0])
        
        # Concatenar
        result = []
        for line in lines:
            result.extend(line)
        
        return result
    
    def _build_phrase_with_spaces(self, chars, boxes):
        """Construye frase con detección de espacios"""
        if len(boxes) <= 1:
            return "".join(chars)
        
        phrase = ""
        char_widths = [w for _, _, w, _ in boxes]
        avg_char_width = np.mean(char_widths)
        
        for i, char in enumerate(chars):
            phrase += char
            
            if i < len(boxes) - 1:
                current_box = boxes[i]
                next_box = boxes[i + 1]
                
                current_right = current_box[0] + current_box[2]
                next_left = next_box[0]
                horizontal_gap = next_left - current_right
                
                current_bottom = current_box[1] + current_box[3]
                next_top = next_box[1]
                vertical_gap = next_top - current_bottom
                
                space_threshold = avg_char_width * 0.7
                avg_height = np.mean([b[3] for b in boxes])
                
                if vertical_gap > avg_height * 0.3:  # Nueva línea
                    phrase += " "
                elif horizontal_gap > space_threshold:  # Espacio
                    phrase += " "
        
        # Limpiar espacios múltiples
        return ' '.join(phrase.split())
    
    def _create_visualization(self, image, boxes, chars, confs, text, content_type):
        """Crea visualización tipo Examen Final"""
        # Convertir a RGB si es necesario
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(image_rgb)
        
        for i, ((x, y, w, h), char, conf) in enumerate(zip(boxes, chars, confs)):
            # Color según confianza
            if conf > 0.8:
                color = 'lime'
            elif conf > 0.5:
                color = 'yellow'
            elif conf > 0.2:
                color = 'orange'
            else:
                color = 'red'
            
            # Dibujar rectángulo
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, 
                edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Texto con predicción
            ax.text(
                x + w//2, y - 5, 
                f"{i+1}: {char}\n{conf:.1%}",
                fontsize=10, fontweight='bold',
                color='blue', ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.2", 
                    facecolor="white", 
                    alpha=0.9
                )
            )
        
        # Título
        emojis = {"LETRA": "🔤", "PALABRA": "📝", "FRASE": "📄"}
        title = f"{emojis[content_type]} Predicción ({content_type}): {text}"
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def show_visualization(self, image_path):
        """Muestra visualización en ventana"""
        text, info, fig = self.predict(
            image_path, debug=True, visualize=True
        )
        plt.show()
        return text, info


def predict_cli(image_path, visualize=False):
    """Predicción desde CLI"""
    predictor = UniversalPredictor()
    
    print(f"\n{'='*60}")
    print(f"🔍 Procesando: {Path(image_path).name}")
    print('='*60)
    
    if visualize:
        text, info = predictor.show_visualization(image_path)
    else:
        text, info = predictor.predict(image_path, debug=True)
    
    print(f"\n{'='*60}")
    print(f"✅ RESULTADO: '{text}'")
    print(f"📊 Tipo: {info['type']} | Caracteres: {info['num_chars']}")
    print('='*60 + "\n")
    
    return text


def predict_with_visualization(image_path):
    """Predicción con visualización"""
    predictor = UniversalPredictor()
    return predictor.show_visualization(image_path)