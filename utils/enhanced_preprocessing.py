"""
Preprocesamiento avanzado con FUSIÓN de puntos para 'i'/'j'
"""

import cv2
import numpy as np
from scipy import ndimage


def smart_resize_for_ocr(image_path):
    """Redimensiona automáticamente para tamaño óptimo de OCR."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    print(f"📐 Imagen original: {w}x{h}")
    
    if w < 200:
        target_width = 300
        scale = target_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"   ↑ Aumentada a: {img.shape[1]}x{img.shape[0]}")
    elif w > 800:
        target_width = 600
        scale = target_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
        print(f"   ↓ Reducida a: {img.shape[1]}x{img.shape[0]}")
    
    return img


def advanced_char_preprocessing(img_array, target_size=(32, 32), debug=False):
    """
    🆕 MEJORADO: Preprocesamiento con mayor contraste y binarización agresiva.
    """
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    original_shape = img_array.shape
    
    if debug:
        print(f"    🔧 ADVANCED PREPROCESSING:")
        print(f"       Original: {original_shape}, rango: [{img_array.min()}, {img_array.max()}]")
    
    # 🔥 MEJORADO: CLAHE más agresivo
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    img_enhanced = clahe.apply(img_array)
    
    if debug:
        print(f"       Post-CLAHE: rango [{img_enhanced.min()}, {img_enhanced.max()}]")
    
    # Detectar si fondo es claro u oscuro
    mean_val = np.mean(img_enhanced)
    if mean_val > 127:
        img_enhanced = 255 - img_enhanced
        if debug:
            print(f"       🔄 INVERTIDO (mean: {mean_val:.1f})")
    
    # 🆕 BINARIZACIÓN AGRESIVA CON OTSU
    _, binary = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 🆕 Eliminar ruido pequeño (morfología)
    kernel_noise = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    # 🆕 Reforzar trazos principales (dilate + erode)
    kernel_thick = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel_thick, iterations=1)
    binary = cv2.erode(binary, kernel_thick, iterations=1)
    
    if debug:
        print(f"       Post-binario: blancos={np.sum(binary>0)}, negros={np.sum(binary==0)}")
    
    # Extraer bounding box del carácter
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        result = np.zeros(target_size, dtype=np.float32)
        if debug:
            print("       ⚠️ Imagen vacía")
        return result
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    char_tight = binary[y_min:y_max+1, x_min:x_max+1]
    
    if debug:
        print(f"       Tight BB: {char_tight.shape}")
    
    # Padding proporcional (20% extra)
    h, w = char_tight.shape
    max_dim = max(h, w)
    padded_size = int(max_dim * 1.2)
    
    pad_h = (padded_size - h) // 2
    pad_w = (padded_size - w) // 2
    
    padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = char_tight
    
    if debug:
        print(f"       Padded: {padded.shape}")
    
    # Redimensionar a 32x32
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    
    # 🆕 NORMALIZACIÓN FUERTE: convertir a 0.0 o 1.0 (sin grises)
    resized_normalized = (resized / 255.0).astype(np.float32)
    resized_normalized = np.where(resized_normalized > 0.5, 1.0, 0.0)
    
    if debug:
        white_pct = (np.sum(resized_normalized > 0.5) / resized_normalized.size) * 100
        print(f"       Final: {resized_normalized.shape}, blancos: {white_pct:.1f}%")
    
    return resized_normalized


def intelligent_segmentation(image_path, debug=False):
    """
    Segmentación mejorada con FUSIÓN de puntos para 'i'/'j'.
    """
    img = smart_resize_for_ocr(image_path)
    h, w = img.shape
    
    variance = np.var(img)
    is_handwritten = variance > 800
    
    if debug:
        print(f"   Tipo: {'Manuscrita' if is_handwritten else 'Digital'} (var={variance:.0f})")
    
    # 🆕 MEJORADO: Preprocesamiento más agresivo
    if is_handwritten:
        # CLAHE más fuerte
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        
        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            blockSize=15, 
            C=8
        )
        
        # Morfología para limpiar
        kernel_clean = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)
        
    else:
        # Para texto digital: Otsu directo
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Reforzar trazos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos
    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        aspect_ratio = cw / ch if ch > 0 else 0
        
        if is_handwritten:
            min_area = 20
            min_dimension = 5
            max_width = w * 0.5
            max_height = h * 0.9
            min_aspect = 0.05
            max_aspect = 6.0
        else:
            min_area = 30
            min_dimension = 8
            max_width = w * 0.3
            max_height = h * 0.8
            min_aspect = 0.08
            max_aspect = 4.0
        
        if (area >= min_area and 
            cw >= min_dimension and ch >= min_dimension and
            cw <= max_width and ch <= max_height and
            min_aspect <= aspect_ratio <= max_aspect):
            boxes.append((x, y, cw, ch))
    
    if debug:
        print(f"   Segmentación inicial: {len(boxes)} caracteres")
    
    # Fusionar puntos con 'i'/'j'
    boxes = merge_dots_with_stems_improved(boxes, h, w, debug=debug)
    
    if debug:
        print(f"   Segmentación final: {len(boxes)} caracteres")
    
    return boxes, img


def merge_dots_with_stems_improved(boxes, image_height, image_width, debug=False):
    """
    Fusiona puntos de 'i'/'j' usando análisis espacial bidimensional.
    """
    if len(boxes) <= 1:
        return boxes
    
    # Clasificar boxes
    dots = []
    stems = []
    others = []
    
    for i, (x, y, w, h) in enumerate(boxes):
        area = w * h
        aspect = w / h if h > 0 else 0
        
        # ¿Es un punto pequeño?
        is_dot = (
            area < 200 and 
            0.4 < aspect < 2.5 and 
            h < image_height * 0.2
        )
        
        # ¿Es una línea vertical?
        is_stem = (
            aspect < 0.6 and 
            h > w * 1.5 and
            h > image_height * 0.3
        )
        
        if is_dot:
            dots.append((i, x, y, w, h))
        elif is_stem:
            stems.append((i, x, y, w, h))
        else:
            others.append((i, x, y, w, h))
    
    if debug:
        print(f"      Clasificación: {len(dots)} puntos, {len(stems)} líneas, {len(others)} otros")
    
    # Fusionar puntos con líneas
    merged_indices = set()
    merged_boxes = []
    
    for dot_idx, dot_x, dot_y, dot_w, dot_h in dots:
        dot_center_x = dot_x + dot_w / 2
        dot_bottom = dot_y + dot_h
        
        best_stem = None
        best_distance = float('inf')
        
        for stem_idx, stem_x, stem_y, stem_w, stem_h in stems:
            if stem_idx in merged_indices:
                continue
            
            stem_center_x = stem_x + stem_w / 2
            stem_top = stem_y
            
            vertical_gap = stem_top - dot_bottom
            if vertical_gap < -10:
                continue
            
            horizontal_distance = abs(dot_center_x - stem_center_x)
            max_horizontal_offset = max(dot_w, stem_w) * 0.6
            max_vertical_gap = image_height * 0.4
            
            if (horizontal_distance < max_horizontal_offset and 
                0 <= vertical_gap < max_vertical_gap):
                
                total_distance = np.sqrt(horizontal_distance**2 + vertical_gap**2)
                
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_stem = (stem_idx, stem_x, stem_y, stem_w, stem_h)
        
        if best_stem:
            stem_idx, stem_x, stem_y, stem_w, stem_h = best_stem
            
            new_x = min(dot_x, stem_x)
            new_y = dot_y
            new_right = max(dot_x + dot_w, stem_x + stem_w)
            new_bottom = max(dot_y + dot_h, stem_y + stem_h)
            new_w = new_right - new_x
            new_h = new_bottom - new_y
            
            merged_boxes.append((new_x, new_y, new_w, new_h))
            merged_indices.add(dot_idx)
            merged_indices.add(stem_idx)
            
            if debug:
                print(f"      🔗 Fusionado: punto({dot_x},{dot_y},{dot_w},{dot_h}) + línea({stem_x},{stem_y},{stem_w},{stem_h})")
    
    # Agregar boxes no fusionados
    all_boxes_indexed = dots + stems + others
    for idx, x, y, w, h in all_boxes_indexed:
        if idx not in merged_indices:
            merged_boxes.append((x, y, w, h))
    
    # Ordenar por posición X
    merged_boxes = sorted(merged_boxes, key=lambda b: b[0])
    
    if debug:
        print(f"      Resultado: {len(merged_boxes)} boxes ({len(merged_indices)} fusionados)")
    
    return merged_boxes