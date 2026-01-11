"""
Funciones de dataset - MAPEO 
"""

import cv2
import numpy as np
from pathlib import Path

# Intentar importar preprocesamiento avanzado
try:
    from utils.enhanced_preprocessing import advanced_char_preprocessing
    USING_ADVANCED = True
    print("✅ Usando enhanced_preprocessing")
except ImportError:
    USING_ADVANCED = False
    print("⚠️  enhanced_preprocessing no disponible, usando versión básica")


# ============================================================================
# LABEL_MAP EXACTO 
# ============================================================================
LABEL_MAP = {
    # Números 0-9 (labels 0-9)
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    
    # Mayúsculas A-Z (labels 10-35)
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    
    # Minúsculas a-z (labels 36-61)
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def preprocess_char_unified(char_img, debug=False):
    """
    Preprocesamiento UNIFICADO - usa advanced si está disponible
    """
    if USING_ADVANCED:
        return advanced_char_preprocessing(char_img, target_size=(32, 32), debug=debug)
    else:
        # Fallback básico
        if debug:
            print(f"    📏 Entrada (básico): {char_img.shape}, rango: [{char_img.min()}, {char_img.max()}]")
        
        if char_img.size == 0:
            return np.zeros((32, 32), dtype=np.float32)
        
        char_resized = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)
        
        mean_val = np.mean(char_resized)
        if mean_val > 127:
            char_resized = 255 - char_resized
            if debug:
                print(f"    🔄 Invertido (mean: {mean_val:.1f})")
        
        char_normalized = char_resized.astype('float32') / 255.0
        
        threshold = 0.15
        char_binary = np.where(char_normalized > threshold, 1.0, 0.0)
        
        if debug:
            white_pct = (np.sum(char_binary == 1.0) / char_binary.size) * 100
            print(f"    📊 Blancos: {white_pct:.1f}%")
        
        return char_binary.astype('float32')


def get_char_from_index(index):
    return LABEL_MAP.get(index, '?')

def get_index_from_char(char):
    return REVERSE_LABEL_MAP.get(char, None)

def validate_label(char):
    return char in REVERSE_LABEL_MAP

def print_label_map():
    print("\n" + "="*60)
    print("📋 LABEL_MAP (62 clases)")
    print("="*60)
    print("\n🔢 NÚMEROS (0-9):")
    print("   " + " ".join([LABEL_MAP[i] for i in range(10)]))
    print("\n🔠 MAYÚSCULAS (10-35):")
    print("   " + " ".join([LABEL_MAP[i] for i in range(10, 36)]))
    print("\n🔡 MINÚSCULAS (36-61):")
    print("   " + " ".join([LABEL_MAP[i] for i in range(36, 62)]))
    
    if USING_ADVANCED:
        print("\n✅ Preprocesamiento: ADVANCED (robusto)")
    else:
        print("\n⚠️  Preprocesamiento: BÁSICO (menos preciso)")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print_label_map()
    
    # Verificar mapeo
    print("🧪 VERIFICACIÓN DE MAPEO:")
    test_cases = [
        (0, '0'), (9, '9'),           # Números
        (10, 'A'), (35, 'Z'),         # Mayúsculas
        (36, 'a'), (61, 'z'),         # Minúsculas
        (18, 'I'), (44, 'i')          # Casos problemáticos I/i
    ]
    
    for idx, expected in test_cases:
        actual = LABEL_MAP.get(idx, '?')
        status = '✅' if actual == expected else '❌'
        print(f"   {status} índice {idx:2d} → '{actual}' (esperado: '{expected}')")