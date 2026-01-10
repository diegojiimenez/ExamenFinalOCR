"""
Preprocesamiento avanzado del proyecto original
"""

import cv2
import numpy as np
from scipy import ndimage


def advanced_char_preprocessing(img_array, target_size=(32, 32), debug=False):
    """
    Preprocesamiento EXTREMADAMENTE robusto para caracteres.
    Copiado EXACTAMENTE del proyecto original.
    """
    # Asegurar escala de grises
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    original_shape = img_array.shape
    
    if debug:
        print(f"    🔧 ADVANCED PREPROCESSING:")
        print(f"       Original: {original_shape}, rango: [{img_array.min()}, {img_array.max()}]")
    
    # 1. Corrección de contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img_enhanced = clahe.apply(img_array)
    
    if debug:
        print(f"       Post-CLAHE: rango [{img_enhanced.min()}, {img_enhanced.max()}]")
    
    # 2. Detección automática de inversión
    mean_val = np.mean(img_enhanced)
    if mean_val > 127:  # Fondo claro
        img_enhanced = 255 - img_enhanced
        if debug:
            print(f"       🔄 INVERTIDO (mean: {mean_val:.1f})")
    
    # 3. Binarización Otsu + limpieza morfológica
    _, binary = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Eliminación de ruido pequeño
    kernel_noise = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    # 5. Encontrar el bounding box del carácter real
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        # Si no hay contenido, crear imagen vacía
        result = np.zeros(target_size, dtype=np.float32)
        if debug:
            print("       ⚠️ Imagen vacía después del preprocesamiento")
        return result
    
    # Calcular bounding box apretado
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extraer solo el carácter
    char_tight = binary[y_min:y_max+1, x_min:x_max+1]
    
    if debug:
        print(f"       Tight BB: {char_tight.shape}")
    
    # 6. Normalización de aspecto con padding inteligente
    h, w = char_tight.shape
    
    # Calcular padding para hacer cuadrado
    max_dim = max(h, w)
    
    # Añadir 20% de padding
    padded_size = int(max_dim * 1.2)
    
    # Centrar el carácter
    pad_h = (padded_size - h) // 2
    pad_w = (padded_size - w) // 2
    
    padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = char_tight
    
    if debug:
        print(f"       Padded: {padded.shape}")
    
    # 7. Redimensionar con interpolación suave
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    
    # 8. Normalización final
    resized_normalized = resized / 255.0
    
    # 9. Mejora final del contraste
    resized_normalized = np.clip(resized_normalized * 1.2, 0, 1)
    
    if debug:
        white_pct = (np.sum(resized_normalized > 0.5) / resized_normalized.size) * 100
        print(f"       Final: {resized_normalized.shape}, blancos: {white_pct:.1f}%")
    
    return resized_normalized.astype(np.float32)


def intelligent_segmentation(image_path, debug=False):
    """
    Segmentación mejorada para manuscritas.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"No se pudo cargar: {image_path}")
    
    h, w = img.shape
    
    # Detectar tipo (manuscrita vs digital)
    variance = np.var(img)
    is_handwritten = variance > 800
    
    if debug:
        print(f"   Tipo: {'Manuscrita' if is_handwritten else 'Digital'} (var={variance:.0f})")
    
    # Preprocesamiento según tipo
    if is_handwritten:
        # Manuscritas: más permisivo
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Digitales: estándar
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtros adaptativos
    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        
        if is_handwritten:
            # Filtros más permisivos para manuscritas
            min_area = 20
            min_dimension = 5
            max_width = w * 0.5
            max_height = h * 0.9
        else:
            # Filtros estrictos para digitales
            min_area = 50
            min_dimension = 8
            max_width = w * 0.3
            max_height = h * 0.8
        
        if (area >= min_area and cw >= min_dimension and ch >= min_dimension and
            cw <= max_width and ch <= max_height):
            boxes.append((x, y, cw, ch))
    
    return boxes, img