"""
Predictor IDÉNTICO al proyecto original
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from core.model_manager import ModelManager
from utils.config import config
from utils.dataset import LABEL_MAP, preprocess_char_unified

# Importar segmentación inteligente si está disponible
try:
    from utils.enhanced_preprocessing import intelligent_segmentation
    USING_INTELLIGENT_SEG = True
except ImportError:
    USING_INTELLIGENT_SEG = False


class UniversalPredictor:
    """Predictor con lógica EXACTA del proyecto original"""
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.get('model.path', 'models/trained_model.h5')
        
        self.model_manager = ModelManager(model_path)
        self.model = self.model_manager.load_model()
        
        self.input_size = 32
        self.num_classes = 62
        
        print(f"📊 Modelo: {self.input_size}x{self.input_size}, {self.num_classes} clases")
        
        if USING_INTELLIGENT_SEG:
            print("✅ Usando intelligent_segmentation (optimizado)")
        else:
            print("⚠️  Usando segmentación básica")
    
    def predict(self, image_path, debug=False, visualize=False):
        """
        Predicción usando INTELLIGENT_SEGMENTATION del proyecto original
        """
        print(f"\n{'='*70}")
        print(f"🔍 PROCESANDO: {Path(image_path).name}")
        print('='*70)
        
        if not Path(image_path).exists():
            raise ValueError(f"❌ No existe: {image_path}")
        
        # USAR INTELLIGENT_SEGMENTATION si está disponible
        if USING_INTELLIGENT_SEG:
            print("\n🔧 Usando intelligent_segmentation...")
            char_boxes, original = intelligent_segmentation(str(image_path), debug=debug)
        else:
            # Fallback: segmentación básica
            print("\n⚠️  Usando segmentación básica (menos precisa)")
            original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise ValueError(f"❌ No se pudo cargar: {image_path}")
            
            char_boxes = self._basic_segmentation(original, debug)
        
        h, w = original.shape
        print(f"📐 Dimensiones: {w}x{h}")
        
        num_chars = len(char_boxes)
        print(f"🔍 Caracteres detectados: {num_chars}")
        
        if not char_boxes:
            return "", {'text': '', 'type': 'VACÍO', 'num_chars': 0}
        
        # Tipo de contenido
        if num_chars == 1:
            content_type = "LETRA"
        elif 2 <= num_chars <= 6:
            content_type = "PALABRA"
        else:
            content_type = "FRASE"
        
        print(f"📋 Tipo detectado: {content_type}")
        
        # Ordenar
        char_boxes = self._smart_sort(char_boxes, content_type)
        
        # RECONOCIMIENTO
        print(f"\n🎯 RECONOCIMIENTO ({num_chars} caracteres):")
        print("-" * 70)
        
        recognized_chars = []
        confidences = []
        
        for i, (x, y, cw, ch) in enumerate(char_boxes):
            # Extraer con margen
            margin = max(2, min(cw, ch) // 8)
            y_start = max(0, y - margin)
            y_end = min(h, y + ch + margin)
            x_start = max(0, x - margin)
            x_end = min(w, x + cw + margin)
            
            char_img = original[y_start:y_end, x_start:x_end]
            
            if debug and i < 3:
                print(f"\n   [{i+1}] Región: ({x_start}, {y_start}) → ({x_end}, {y_end})")
                print(f"       Tamaño extraído: {char_img.shape}")
            
            # Preprocesar (usa advanced_char_preprocessing automáticamente)
            char_processed = preprocess_char_unified(char_img, debug=(debug and i < 3))
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
                char_alt = (char_alt / 255.0).astype('float32')
                char_alt = np.where(char_alt > 0.15, 1.0, 0.0)
                
                char_alt_input = char_alt.reshape(1, 32, 32, 1)
                prediction_alt = self.model.predict(char_alt_input, verbose=0)
                confidence_alt = np.max(prediction_alt)
                
                if confidence_alt > confidence:
                    prediction = prediction_alt
                    predicted_class = np.argmax(prediction)
                    confidence = confidence_alt
                    if debug and i < 3:
                        print(f"       🔄 Preprocesamiento alternativo")
            
            # Decodificar
            predicted_char = LABEL_MAP.get(predicted_class, '?')
            recognized_chars.append(predicted_char)
            confidences.append(confidence)
            
            # Mostrar resultado
            color = '🟢' if confidence > 0.8 else ('🟡' if confidence > 0.5 else '🔴')
            print(f"   {color} [{i+1:2d}] '{predicted_char}' ({confidence:.1%}) | clase={predicted_class}")
            
            if debug and i < 3:
                top3_indices = np.argsort(prediction[0])[-3:][::-1]
                print(f"       Top 3: ", end="")
                for idx in top3_indices:
                    char = LABEL_MAP.get(idx, '?')
                    conf = prediction[0][idx]
                    print(f"'{char}':{conf:.1%} ", end="")
                print()
        
        # Construir texto
        if content_type == "LETRA":
            phrase = "".join(recognized_chars)
        elif content_type == "PALABRA":
            phrase = "".join(recognized_chars)
        else:
            phrase = self._build_phrase_with_spaces(recognized_chars, char_boxes)
        
        print("-" * 70)
        print(f"📝 RESULTADO FINAL: '{phrase}'")
        print('='*70)
        
        info = {
            'text': phrase,
            'type': content_type,
            'num_chars': num_chars,
            'boxes': char_boxes,
            'confidences': confidences,
            'recognized_chars': recognized_chars
        }
        
        if visualize:
            fig = self._create_modern_visualization(
                original, char_boxes, recognized_chars,
                confidences, phrase, content_type
            )
            return phrase, info, fig
        
        return phrase, info
    
    def _basic_segmentation(self, original, debug):
        """Segmentación básica (fallback)"""
        variance = np.var(original)
        is_handwritten = variance > 800
        
        if is_handwritten:
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
            _, binary = cv2.threshold(
                original, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        h, w = original.shape
        char_boxes = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            aspect_ratio = cw / ch if ch > 0 else 0
            
            if is_handwritten:
                min_area, min_dim = 20, 5
                max_w, max_h = w * 0.6, h * 0.9
                aspect_range = (0.05, 6.0)
            else:
                min_area, min_dim = 50, 8
                max_w, max_h = w * 0.3, h * 0.8
                aspect_range = (0.1, 3.0)
            
            if (area >= min_area and
                cw >= min_dim and ch >= min_dim and
                cw <= max_w and ch <= max_h and
                aspect_range[0] <= aspect_ratio <= aspect_range[1]):
                char_boxes.append((x, y, cw, ch))
        
        return char_boxes
    
    def _smart_sort(self, boxes, content_type):
        """Ordenamiento inteligente"""
        if not boxes or content_type == "LETRA":
            return boxes
        
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
        
        lines.sort(key=lambda line: np.mean([b[1] for b in line]))
        for line in lines:
            line.sort(key=lambda b: b[0])
        
        result = []
        for line in lines:
            result.extend(line)
        
        return result
    
    def _build_phrase_with_spaces(self, chars, boxes):
        """Construcción de frase con espacios"""
        if len(boxes) <= 1:
            return "".join(chars)
        
        phrase = ""
        avg_char_width = np.mean([w for _, _, w, _ in boxes])
        
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
                
                if vertical_gap > avg_height * 0.3:
                    phrase += " "
                elif horizontal_gap > space_threshold:
                    phrase += " "
        
        return ' '.join(phrase.split())
    
    def _create_modern_visualization(self, image, boxes, chars, confs, text, content_type):
        """Visualización moderna"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig = plt.figure(figsize=(18, 10), facecolor='#1e1e1e')
        ax = plt.subplot(111, facecolor='#2d2d30')
        
        ax.imshow(image_rgb, cmap='gray')
        
        colors = {
            'high': '#00ff88',
            'medium': '#ffd700',
            'low': '#ff6b35',
            'verylow': '#ff0055'
        }
        
        for i, ((x, y, w, h), char, conf) in enumerate(zip(boxes, chars, confs)):
            if conf > 0.8:
                color = colors['high']
            elif conf > 0.5:
                color = colors['medium']
            elif conf > 0.2:
                color = colors['low']
            else:
                color = colors['verylow']
            
            fancy_box = FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=2",
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                alpha=0.9
            )
            ax.add_patch(fancy_box)
            
            label_text = f"#{i+1}\n'{char}'\n{conf:.0%}"
            
            ax.text(
                x + w/2, y - 15,
                label_text,
                fontsize=11,
                fontweight='bold',
                color='white',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=color,
                    edgecolor='white',
                    linewidth=2,
                    alpha=0.95
                ),
                zorder=100
            )
        
        emojis = {"LETRA": "🔤", "PALABRA": "📝", "FRASE": "📄"}
        title = f"{emojis[content_type]} RECONOCIMIENTO OCR | Tipo: {content_type}\nTexto: {text}"
        
        ax.set_title(
            title,
            fontsize=20,
            fontweight='bold',
            color='#00d9ff',
            pad=25,
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor='#1e1e1e',
                edgecolor='#00d9ff',
                linewidth=3
            )
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def show_visualization(self, image_path):
        """Muestra visualización"""
        text, info, fig = self.predict(image_path, debug=True, visualize=True)
        plt.show()
        return text, info


def predict_cli(image_path, visualize=False):
    """CLI"""
    predictor = UniversalPredictor()
    
    if visualize:
        text, info = predictor.show_visualization(image_path)
    else:
        text, info = predictor.predict(image_path, debug=True)
    
    return text


def predict_with_visualization(image_path):
    """Predicción con visualización"""
    predictor = UniversalPredictor()
    return predictor.show_visualization(image_path)