"""
Segmentación de texto en líneas, palabras y caracteres
"""

import cv2
import numpy as np


class TextSegmenter:
    """
    Segmenta texto usando proyecciones y componentes conectados.
    """
    
    def __init__(self, config):
        self.config = config
    
    def segment(self, binary_image):
        """
        Segmenta la imagen en líneas, palabras y caracteres.
        
        Args:
            binary_image (numpy.ndarray): Imagen binarizada
            
        Returns:
            dict: Estructura con segmentos
        """
        lines = self._segment_lines(binary_image)
        
        result = {
            'lines': [],
            'words': [],
            'characters': []
        }
        
        for line_idx, line_img in enumerate(lines):
            words = self._segment_words(line_img)
            
            line_data = {
                'index': line_idx,
                'image': line_img,
                'words': []
            }
            
            for word_idx, word_img in enumerate(words):
                characters = self._segment_characters(word_img)
                
                word_data = {
                    'line_index': line_idx,
                    'word_index': word_idx,
                    'image': word_img,
                    'characters': characters
                }
                
                line_data['words'].append(word_data)
                result['words'].append(word_data)
                result['characters'].extend(characters)
            
            result['lines'].append(line_data)
        
        return result
    
    def _segment_lines(self, image):
        """
        Segmenta líneas usando proyección horizontal.
        """
        # Proyección horizontal: suma de píxeles negros por fila
        h_projection = np.sum(image == 0, axis=1)
        
        # Umbral para detectar líneas (ajustable)
        threshold = np.mean(h_projection) * 0.1
        
        # Encontrar regiones con texto
        in_line = False
        lines = []
        start_row = 0
        
        for i, val in enumerate(h_projection):
            if not in_line and val > threshold:
                in_line = True
                start_row = i
            elif in_line and val < threshold:
                # Verificar que la línea tenga altura mínima
                if i - start_row > self.config['segmentation']['line']['min_height']:
                    lines.append(image[start_row:i, :])
                in_line = False
        
        # Última línea si termina en texto
        if in_line and len(image) - start_row > self.config['segmentation']['line']['min_height']:
            lines.append(image[start_row:, :])
        
        return lines
    
    def _segment_words(self, line_image):
        """
        Segmenta palabras usando proyección vertical.
        """
        # Proyección vertical: suma de píxeles negros por columna
        v_projection = np.sum(line_image == 0, axis=0)
        
        # Detectar espacios entre palabras
        min_gap = self.config['segmentation']['word']['min_gap']
        
        in_word = False
        words = []
        start_col = 0
        gap_count = 0
        
        for i, val in enumerate(v_projection):
            if val == 0:
                gap_count += 1
            else:
                if gap_count >= min_gap and in_word:
                    # Fin de palabra
                    words.append(line_image[:, start_col:i-gap_count])
                    in_word = False
                
                if not in_word:
                    in_word = True
                    start_col = i
                
                gap_count = 0
        
        # Última palabra
        if in_word:
            words.append(line_image[:, start_col:])
        
        return words
    
    def _segment_characters(self, word_image):
        """
        Segmenta caracteres usando componentes conectados.
        """
        # Encontrar componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(word_image),
            connectivity=8
        )
        
        characters = []
        min_width = self.config['segmentation']['letter']['min_width']
        min_height = self.config['segmentation']['letter']['min_height']
        padding = self.config['segmentation']['letter']['padding']
        
        # Ordenar por posición horizontal (izquierda a derecha)
        components = []
        for i in range(1, num_labels):  # Saltar el fondo (0)
            x, y, w, h, area = stats[i]
            
            # Filtrar componentes muy pequeños (ruido)
            if w >= min_width and h >= min_height:
                components.append((x, y, w, h))
        
        components.sort(key=lambda c: c[0])  # Ordenar por x
        
        # Extraer caracteres con padding
        for x, y, w, h in components:
            # Añadir padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(word_image.shape[1], x + w + padding)
            y2 = min(word_image.shape[0], y + h + padding)
            
            char_img = word_image[y1:y2, x1:x2]
            
            # Redimensionar a tamaño fijo para el modelo
            target_size = (28, 28)  # O 32x32 según tu config
            char_resized = cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)
            
            characters.append({
                'image': char_resized,
                'bbox': (x, y, w, h),
                'position': x
            })
        
        return characters