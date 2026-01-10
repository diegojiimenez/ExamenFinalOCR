"""
Funciones de dataset adaptadas del proyecto original "Examen Final"
Incluye LABEL_MAP y preprocesamiento unificado de caracteres
"""

import cv2
import numpy as np
from pathlib import Path


# ============================================================================
# LABEL_MAP: Mapeo de índices de clase a caracteres (62 clases)
# ============================================================================
# 0-25:  A-Z (mayúsculas)
# 26-51: a-z (minúsculas)
# 52-61: 0-9 (dígitos)

LABEL_MAP = {
    **{i: chr(65 + i) for i in range(26)},          # A-Z
    **{i + 26: chr(97 + i) for i in range(26)},     # a-z
    **{i + 52: str(i - 52) for i in range(52, 62)}  # 0-9
}

# Crear mapeo inverso (carácter -> índice)
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def preprocess_char_unified(char_img, debug=False):
    """
    Preprocesamiento unificado para caracteres individuales.
    Adaptado del proyecto original - trabaja con imágenes 32x32.
    
    Args:
        char_img (np.ndarray): Imagen del carácter en escala de grises
        debug (bool): Si True, imprime información de debug
    
    Returns:
        np.ndarray: Imagen preprocesada (32, 32) normalizada [0, 1]
    """
    if debug:
        print(f"    📏 Tamaño entrada: {char_img.shape}")
        print(f"    📊 Rango original: [{char_img.min()}, {char_img.max()}]")
    
    # 1. Verificar que la imagen no esté vacía
    if char_img.size == 0:
        if debug:
            print("    ⚠️  Imagen vacía, retornando imagen en blanco")
        return np.zeros((32, 32), dtype=np.float32)
    
    # 2. Redimensionar a 32x32 (IMPORTANTE: el modelo espera 32x32)
    char_resized = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)
    
    # 3. Detectar y corregir inversión de colores
    # Si el fondo es claro (valor medio alto), invertir
    mean_val = np.mean(char_resized)
    if mean_val > 127:  # Fondo claro, texto oscuro
        char_resized = 255 - char_resized
        if debug:
            print(f"    🔄 Colores invertidos (mean: {mean_val:.1f})")
    
    # 4. Normalizar a rango [0, 1]
    char_normalized = char_resized.astype('float32') / 255.0
    
    # 5. Binarización suave con umbral adaptativo
    # Umbral más bajo = más permisivo (captura más detalles)
    threshold = 0.15
    char_binary = np.where(char_normalized > threshold, 1.0, 0.0)
    
    # 6. Aplicar filtro de suavizado para reducir ruido
    # (Opcional, mejora resultados en imágenes ruidosas)
    char_final = char_binary.astype('float32')
    
    if debug:
        print(f"    📐 Tamaño salida: {char_final.shape}")
        print(f"    📊 Rango valores: [{char_final.min():.3f}, {char_final.max():.3f}]")
        print(f"    ⚫ Píxeles negros (0.0): {np.sum(char_final == 0.0)}")
        print(f"    ⚪ Píxeles blancos (1.0): {np.sum(char_final == 1.0)}")
        
        # Calcular densidad (% de píxeles blancos)
        total_pixels = char_final.size
        white_pixels = np.sum(char_final == 1.0)
        density = (white_pixels / total_pixels) * 100
        print(f"    📈 Densidad: {density:.1f}%")
    
    return char_final


def preprocess_char_with_padding(char_img, target_size=32, debug=False):
    """
    Preprocesamiento con padding para mantener proporción del carácter.
    Útil para caracteres con aspect ratio extremo (muy anchos o muy altos).
    
    Args:
        char_img (np.ndarray): Imagen del carácter
        target_size (int): Tamaño objetivo (default: 32)
        debug (bool): Modo debug
    
    Returns:
        np.ndarray: Imagen preprocesada con padding
    """
    if char_img.size == 0:
        return np.zeros((target_size, target_size), dtype=np.float32)
    
    h, w = char_img.shape
    max_dim = max(h, w)
    
    # Crear imagen cuadrada con padding negro
    squared = np.zeros((max_dim, max_dim), dtype=np.uint8)
    
    # Centrar el carácter
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    squared[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
    
    if debug:
        print(f"    🔲 Padding aplicado: ({h}, {w}) → ({max_dim}, {max_dim})")
    
    # Aplicar preprocesamiento estándar
    return preprocess_char_unified(squared, debug=debug)


def get_char_from_index(index):
    """
    Obtiene el carácter correspondiente a un índice de clase.
    
    Args:
        index (int): Índice de clase (0-61)
    
    Returns:
        str: Carácter correspondiente o '?' si no existe
    """
    return LABEL_MAP.get(index, '?')


def get_index_from_char(char):
    """
    Obtiene el índice de clase correspondiente a un carácter.
    
    Args:
        char (str): Carácter (A-Z, a-z, 0-9)
    
    Returns:
        int: Índice de clase o None si no existe
    """
    return REVERSE_LABEL_MAP.get(char, None)


def validate_label(char):
    """
    Valida si un carácter está en el conjunto de clases soportadas.
    
    Args:
        char (str): Carácter a validar
    
    Returns:
        bool: True si es válido, False en caso contrario
    """
    return char in REVERSE_LABEL_MAP


def print_label_map():
    """
    Imprime el mapa de etiquetas de forma legible.
    Útil para debugging.
    """
    print("\n" + "="*60)
    print("📋 LABEL_MAP (62 clases)")
    print("="*60)
    
    # Mayúsculas
    print("\n🔠 MAYÚSCULAS (0-25):")
    uppercase_chars = [LABEL_MAP[i] for i in range(26)]
    print("   " + " ".join(uppercase_chars))
    
    # Minúsculas
    print("\n🔡 MINÚSCULAS (26-51):")
    lowercase_chars = [LABEL_MAP[i] for i in range(26, 52)]
    print("   " + " ".join(lowercase_chars))
    
    # Números
    print("\n🔢 NÚMEROS (52-61):")
    digit_chars = [LABEL_MAP[i] for i in range(52, 62)]
    print("   " + " ".join(digit_chars))
    
    print("\n" + "="*60 + "\n")


# ============================================================================
# FUNCIÓN DE PRUEBA
# ============================================================================

def test_preprocessing(image_path):
    """
    Función de prueba para verificar el preprocesamiento.
    
    Args:
        image_path (str): Ruta de una imagen de prueba
    """
    import matplotlib.pyplot as plt
    
    print(f"\n🧪 Probando preprocesamiento en: {image_path}")
    
    # Cargar imagen
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ No se pudo cargar la imagen")
        return
    
    print(f"📐 Imagen original: {img.shape}")
    
    # Preprocesar
    processed = preprocess_char_unified(img, debug=True)
    
    # Visualizar
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title("Preprocesada (32x32)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Preprocesamiento completado")


# ============================================================================
# EJECUTAR COMO SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Prueba las funciones del módulo
    """
    print("\n" + "🔤"*30)
    print("MÓDULO DATASET - PRUEBAS")
    print("🔤"*30 + "\n")
    
    # Mostrar label map
    print_label_map()
    
    # Pruebas de conversión
    print("🧪 PRUEBAS DE CONVERSIÓN:")
    print("="*60)
    
    test_chars = ['A', 'Z', 'a', 'z', '0', '9']
    for char in test_chars:
        idx = get_index_from_char(char)
        print(f"   '{char}' → índice: {idx}")
    
    test_indices = [0, 25, 26, 51, 52, 61]
    for idx in test_indices:
        char = get_char_from_index(idx)
        print(f"   índice {idx} → '{char}'")
    
    print("\n" + "="*60)
    print("✅ Módulo dataset cargado correctamente")
    print("="*60 + "\n")