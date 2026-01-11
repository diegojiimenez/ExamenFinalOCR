"""
Post-corrección basada en ANÁLISIS GEOMÉTRICO (sin diccionario)
"""

import cv2
import numpy as np


class OCRPostCorrector:
    """
    Corrige errores usando ANÁLISIS GEOMÉTRICO de los caracteres.
    No usa diccionario, solo geometría y contexto.
    """
    
    def __init__(self):
        pass
    
    def correct_text(self, text, char_list, confidences, char_images=None, debug=False):
        """
        Corrige texto usando análisis geométrico
        
        Args:
            text: Texto predicho
            char_list: Lista de caracteres
            confidences: Confianzas
            char_images: Lista de imágenes de cada carácter (opcional pero recomendado)
            debug: Debug
        
        Returns:
            Texto corregido
        """
        if not char_list or not confidences:
            return text
        
        if debug:
            print("\n🔧 POST-CORRECCIÓN GEOMÉTRICA:")
            print(f"   Original: '{text}'")
        
        corrected_chars = char_list.copy()
        corrections_made = []
        
        # Analizar cada carácter con baja confianza
        for i, (char, conf) in enumerate(zip(char_list, confidences)):
            
            # Solo procesar caracteres ambiguos con confianza < 60%
            if conf >= 0.6:
                continue
            
            # Analizar según el carácter
            if char in ['P', 'p']:  # Confusión común con 'I'/'i'
                corrected = self._analyze_p_vs_i(char, i, char_list, char_images, debug)
                if corrected != char:
                    corrected_chars[i] = corrected
                    corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['l', '1', '|']:  # Confusión con 'I'
                corrected = self._analyze_thin_chars(char, i, char_list, char_images, debug)
                if corrected != char:
                    corrected_chars[i] = corrected
                    corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['I', 'i']:  # Validar si realmente es 'I'
                corrected = self._validate_i(char, i, char_list, char_images, debug)
                if corrected != char:
                    corrected_chars[i] = corrected
                    corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['O', '0', 'o']:  # Confusión O/0
                corrected = self._analyze_o_vs_zero(char, i, char_list, debug)
                if corrected != char:
                    corrected_chars[i] = corrected
                    corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
        
        corrected_text = ''.join(corrected_chars)
        
        if debug and corrections_made:
            print(f"   Correcciones aplicadas:")
            for corr in corrections_made:
                print(f"      - {corr}")
            print(f"   Resultado: '{corrected_text}'")
        
        return corrected_text
    
    def _analyze_p_vs_i(self, char, index, char_list, char_images, debug):
        """
        Distingue entre 'P' y 'I' usando geometría
        
        Características clave:
        - 'P': Tiene bucle cerrado en la parte superior
        - 'I': Es delgada y sin bucles
        """
        context = self._get_context(char_list, index)
        
        # Regla 1: Contexto alfabético
        # Si está entre letras mayúsculas Y tiene confianza baja, podría ser 'I'
        prev_is_upper = context['prev'] and context['prev'].isupper()
        next_is_upper = context['next'] and context['next'].isupper()
        
        if prev_is_upper and next_is_upper:
            if debug:
                print(f"      [{index}] '{char}' entre mayúsculas → probablemente 'I'")
            return 'I'
        
        # Regla 2: Si está entre minúsculas, podría ser 'i'
        prev_is_lower = context['prev'] and context['prev'].islower()
        next_is_lower = context['next'] and context['next'].islower()
        
        if prev_is_lower and next_is_lower and char == 'p':
            if debug:
                print(f"      [{index}] 'p' entre minúsculas → probablemente 'i'")
            return 'i'
        
        # Regla 3: Análisis geométrico (si tenemos imagen)
        if char_images and index < len(char_images):
            img = char_images[index]
            
            # Calcular aspect ratio
            h, w = img.shape[:2] if len(img.shape) > 1 else (img.shape[0], 1)
            aspect_ratio = w / h if h > 0 else 0
            
            # 'I' es muy delgada (aspect ratio < 0.4)
            # 'P' tiene más ancho (aspect ratio > 0.4)
            if aspect_ratio < 0.35:
                if debug:
                    print(f"      [{index}] Aspect ratio {aspect_ratio:.2f} → muy delgada → 'I'")
                return 'I' if char.isupper() else 'i'
        
        return char
    
    def _analyze_thin_chars(self, char, index, char_list, char_images, debug):
        """
        Analiza caracteres delgados: 'l', '1', '|' vs 'I'/'i'
        """
        context = self._get_context(char_list, index)
        
        # Si está rodeado de letras mayúsculas → 'I'
        if context['prev'] and context['prev'].isupper() and context['next'] and context['next'].isupper():
            if debug:
                print(f"      [{index}] '{char}' entre mayúsculas → 'I'")
            return 'I'
        
        # Si está rodeado de letras minúsculas → 'i'
        if context['prev'] and context['prev'].islower() and context['next'] and context['next'].islower():
            if debug:
                print(f"      [{index}] '{char}' entre minúsculas → 'i'")
            return 'i'
        
        # Al inicio de palabra → mayúscula
        if context['position'] == 'start' or (context['prev'] and context['prev'] == ' '):
            if debug:
                print(f"      [{index}] '{char}' al inicio → 'I'")
            return 'I'
        
        return char
    
    def _validate_i(self, char, index, char_list, char_images, debug):
        """
        Valida si un carácter clasificado como 'I'/'i' es correcto
        """
        context = self._get_context(char_list, index)
        
        # Si 'I' está entre minúsculas, probablemente sea 'i'
        if char == 'I':
            prev_is_lower = context['prev'] and context['prev'].islower()
            next_is_lower = context['next'] and context['next'].islower()
            
            if prev_is_lower and next_is_lower:
                if debug:
                    print(f"      [{index}] 'I' entre minúsculas → 'i'")
                return 'i'
        
        # Si 'i' está entre mayúsculas, probablemente sea 'I'
        elif char == 'i':
            prev_is_upper = context['prev'] and context['prev'].isupper()
            next_is_upper = context['next'] and context['next'].isupper()
            
            if prev_is_upper and next_is_upper:
                if debug:
                    print(f"      [{index}] 'i' entre mayúsculas → 'I'")
                return 'I'
        
        return char
    
    def _analyze_o_vs_zero(self, char, index, char_list, debug):
        """
        Distingue entre 'O'/'o' y '0'
        """
        context = self._get_context(char_list, index)
        
        # Si está rodeado de letras → probablemente sea 'O'/'o'
        prev_is_alpha = context['prev'] and context['prev'].isalpha()
        next_is_alpha = context['next'] and context['next'].isalpha()
        
        if prev_is_alpha and next_is_alpha:
            if char == '0':
                if context['prev'] and context['prev'].isupper():
                    if debug:
                        print(f"      [{index}] '0' entre letras mayúsculas → 'O'")
                    return 'O'
                else:
                    if debug:
                        print(f"      [{index}] '0' entre letras minúsculas → 'o'")
                    return 'o'
        
        # Si está rodeado de números → probablemente sea '0'
        prev_is_digit = context['prev'] and context['prev'].isdigit()
        next_is_digit = context['next'] and context['next'].isdigit()
        
        if prev_is_digit or next_is_digit:
            if char in ['O', 'o']:
                if debug:
                    print(f"      [{index}] '{char}' cerca de números → '0'")
                return '0'
        
        return char
    
    def _get_context(self, char_list, index):
        """Obtiene el contexto de un carácter"""
        return {
            'prev': char_list[index-1] if index > 0 else None,
            'next': char_list[index+1] if index < len(char_list)-1 else None,
            'position': 'start' if index == 0 else ('end' if index == len(char_list)-1 else 'middle')
        }


# Instancia global
post_corrector = OCRPostCorrector()