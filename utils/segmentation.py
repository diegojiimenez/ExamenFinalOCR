"""
Segmentación de caracteres en imágenes
"""

import cv2
import numpy as np


def segment_characters(binary_image, debug=False):
    """
    Segmenta caracteres individuales de una imagen
    
    Args:
        binary_image: Imagen binarizada
        debug: Mostrar información de debug
        
    Returns:
        Lista de bounding boxes [(x, y, w, h), ...]
    """
    # Encontrar contornos
    contours, _ = cv2.findContours(
        binary_image, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filtrar contornos pequeños (ruido)
    min_area = 50
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Extraer bounding boxes
    boxes = []
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filtrar por dimensiones razonables
        if w > 5 and h > 10:
            boxes.append((x, y, w, h))
    
    # Ordenar de izquierda a derecha
    boxes = sorted(boxes, key=lambda b: b[0])
    
    if debug:
        print(f"  Contornos totales: {len(contours)}")
        print(f"  Contornos válidos: {len(valid_contours)}")
        print(f"  Caracteres finales: {len(boxes)}")
    
    return boxes


def visualize_boxes(image, boxes, output_path="debug_boxes.png"):
    """Dibuja los bounding boxes sobre la imagen"""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, vis)
    print(f"💾 Visualización guardada en: {output_path}")