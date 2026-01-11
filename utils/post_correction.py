"""
Post-corrección basada en ANÁLISIS GEOMÉTRICO PURO
Enfocada en el problema general de 'I'/'i', no en palabras específicas
"""

import cv2
import numpy as np


class OCRPostCorrector:
    """
    Corrige 'I'/'i' usando SOLO geometría, sin depender del contexto.
    """
    
    def __init__(self):
        pass
    
    def correct_text(self, text, char_list, confidences, char_images=None, debug=False):
        """
        Corrige caracteres confundibles analizando SOLO su geometría
        """
        if not char_list or not confidences:
            return text
        
        if debug:
            print("\n🔧 POST-CORRECCIÓN GEOMÉTRICA:")
            print(f"   Original: '{text}'")
        
        corrected_chars = char_list.copy()
        corrections_made = []
        
        # Determinar el "modo" de la frase (mayúsculas vs minúsculas)
        uppercase_count = sum(1 for c in char_list if c.isupper())
        lowercase_count = sum(1 for c in char_list if c.islower())
        total_letters = uppercase_count + lowercase_count
        
        # 🔧 FIX: Si no hay letras, no aplicar corrección de modo
        if total_letters == 0:
            if debug:
                print(f"   Modo: SIN LETRAS (solo números/símbolos) - No aplicar corrección")
            return text  # ✅ No corregir si solo hay números
        
        # Si predominan mayúsculas → modo mayúscula, si predominan minúsculas → modo minúscula
        is_uppercase_mode = uppercase_count > lowercase_count
        
        if debug:
            print(f"   Modo detectado: {'MAYÚSCULAS' if is_uppercase_mode else 'minúsculas'}")
            print(f"   (May: {uppercase_count}, Min: {lowercase_count})")
        
        # Analizar cada carácter
        for i, (char, conf) in enumerate(zip(char_list, confidences)):
            
            # 🔧 FIX CRÍTICO: Solo analizar caracteres con BAJA confianza O específicos problemáticos
            
            if char in ['P', 'p']:
                # Solo analizar 'P'/'p' si confianza < 50% (muy baja)
                if conf < 0.50:
                    corrected = self._analyze_thin_character(
                        char, i, char_list, char_images, 
                        confidences[i], is_uppercase_mode, debug
                    )
                    if corrected != char:
                        corrected_chars[i] = corrected
                        corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char == 'f':
                # 'f' es muy problemática, analizar hasta 85% de confianza
                if conf < 0.85:
                    corrected = self._analyze_thin_character(
                        char, i, char_list, char_images, 
                        confidences[i], is_uppercase_mode, debug
                    )
                    if corrected != char:
                        corrected_chars[i] = corrected
                        corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['l', '1', '|']:
                # 🔧 FIX: Umbral MUY BAJO para '1' - solo si confianza < 40%
                # Si '1' tiene >40% confianza, probablemente ES un número
                if conf < 0.40:  # ✅ Bajado de 0.70 a 0.40
                    # Verificar si realmente debería ser letra
                    if self._should_be_letter(char, i, char_list, debug):
                        corrected = self._analyze_thin_character(
                            char, i, char_list, char_images, 
                            confidences[i], is_uppercase_mode, debug
                        )
                        if corrected != char:
                            corrected_chars[i] = corrected
                            corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            # Validar 'I'/'i' ya reconocidas
            elif char in ['I', 'i'] and conf < 0.85:
                corrected = self._validate_i_character(
                    char, i, char_list, is_uppercase_mode, debug
                )
                if corrected != char:
                    corrected_chars[i] = corrected
                    corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            # Analizar O/0
            elif char in ['O', '0', 'o'] and conf < 0.70:
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
        elif debug:
            print(f"   Sin correcciones")
        
        return corrected_text
    
    def _should_be_letter(self, char, index, char_list, debug):
        """
        🆕 Verifica si un carácter '1'/'l'/'|' debería ser letra o número
        
        Reglas:
        - Si está rodeado de LETRAS → probablemente sea letra ('I' o 'i')
        - Si está rodeado de NÚMEROS → probablemente sea número ('1')
        - Si está solo → no cambiar
        """
        context = self._get_context(char_list, index)
        
        # Contar caracteres alfanuméricos vecinos
        prev_is_letter = context['prev'] and context['prev'].isalpha()
        next_is_letter = context['next'] and context['next'].isalpha()
        
        prev_is_digit = context['prev'] and context['prev'].isdigit()
        next_is_digit = context['next'] and context['next'].isdigit()
        
        # Si está entre LETRAS → debería ser letra
        if prev_is_letter or next_is_letter:
            if debug:
                print(f"         → Cerca de letras, podría ser 'I'/'i'")
            return True
        
        # Si está entre NÚMEROS → es número
        if prev_is_digit or next_is_digit:
            if debug:
                print(f"         → Cerca de números, es '1'")
            return False
        
        # Si está solo o sin contexto claro → NO cambiar (es número)
        if debug:
            print(f"         → Sin contexto claro, mantener como número")
        return False
    
    def _analyze_thin_character(self, char, index, char_list, char_images, 
                                confidence, is_uppercase_mode, debug):
        """
        Analiza UN carácter delgado que podría ser 'I' o 'i'
        
        Regla simple:
        1. Si aspect ratio < 0.35 → es 'I' o 'i' (depende del modo)
        2. Si confianza < 50% Y está solo → probablemente sea 'I'/'i'
        """
        
        # Análisis geométrico
        if char_images and index < len(char_images):
            img = char_images[index]
            h, w = img.shape[:2] if len(img.shape) > 1 else (img.shape[0], 1)
            aspect_ratio = w / h if h > 0 else 0
            
            if debug:
                print(f"      [{index}] '{char}' | AR: {aspect_ratio:.2f} | Conf: {confidence:.1%}")
            
            # 🔧 REGLA PRINCIPAL: Si es MUY delgado → es 'I' o 'i'
            if aspect_ratio < 0.35:
                # Decidir entre 'I' o 'i' basándose en el MODO de la frase
                result = 'I' if is_uppercase_mode else 'i'
                
                if debug:
                    print(f"         → AR < 0.35, modo {('MAY' if is_uppercase_mode else 'min')} → '{result}'")
                
                return result
            
            # Si aspect ratio ambiguo (0.35-0.50) Y confianza baja
            elif aspect_ratio < 0.50 and confidence < 0.50:
                result = 'I' if is_uppercase_mode else 'i'
                
                if debug:
                    print(f"         → AR ambiguo + baja conf → '{result}'")
                
                return result
        
        # Sin imagen o aspect ratio normal → analizar contexto local
        context = self._get_context(char_list, index)
        
        # Si el carácter anterior/siguiente es mayúscula → probablemente 'I'
        if context['prev'] and context['prev'].isupper():
            if debug:
                print(f"         → Después de mayúscula → 'I'")
            return 'I'
        
        if context['next'] and context['next'].isupper():
            if debug:
                print(f"         → Antes de mayúscula → 'I'")
            return 'I'
        
        # Si el carácter anterior/siguiente es minúscula → probablemente 'i'
        if context['prev'] and context['prev'].islower():
            if debug:
                print(f"         → Después de minúscula → 'i'")
            return 'i'
        
        if context['next'] and context['next'].islower():
            if debug:
                print(f"         → Antes de minúscula → 'i'")
            return 'i'
        
        # Por defecto, usar el modo de la frase
        return 'I' if is_uppercase_mode else 'i'
    
    def _validate_i_character(self, char, index, char_list, is_uppercase_mode, debug):
        """
        Valida si 'I' o 'i' reconocida es correcta
        """
        context = self._get_context(char_list, index)
        
        # Si 'I' está entre minúsculas → cambiar a 'i'
        if char == 'I':
            if context['prev'] and context['prev'].islower() and context['next'] and context['next'].islower():
                if debug:
                    print(f"      [{index}] 'I' entre minúsculas → 'i'")
                return 'i'
        
        # Si 'i' está entre mayúsculas → cambiar a 'I'
        elif char == 'i':
            if context['prev'] and context['prev'].isupper() and context['next'] and context['next'].isupper():
                if debug:
                    print(f"      [{index}] 'i' entre mayúsculas → 'I'")
                return 'I'
        
        return char
    
    def _analyze_o_vs_zero(self, char, index, char_list, debug):
        """
        Distingue entre 'O'/'o' y '0'
        """
        context = self._get_context(char_list, index)
        
        # Si está rodeado de letras → 'O'/'o'
        prev_is_alpha = context['prev'] and context['prev'].isalpha()
        next_is_alpha = context['next'] and context['next'].isalpha()
        
        if prev_is_alpha or next_is_alpha:
            if char == '0':
                # Decidir entre 'O' o 'o' por el contexto
                if context['prev'] and context['prev'].isupper():
                    if debug:
                        print(f"      [{index}] '0' entre letras MAY → 'O'")
                    return 'O'
                else:
                    if debug:
                        print(f"      [{index}] '0' entre letras min → 'o'")
                    return 'o'
        
        # Si está cerca de números → '0'
        prev_is_digit = context['prev'] and context['prev'].isdigit()
        next_is_digit = context['next'] and context['next'].isdigit()
        
        if prev_is_digit or next_is_digit:
            if char in ['O', 'o']:
                if debug:
                    print(f"      [{index}] '{char}' cerca de números → '0'")
                return '0'
        
        return char
    
    def _get_context(self, char_list, index):
        """Obtiene contexto del carácter"""
        return {
            'prev': char_list[index-1] if index > 0 else None,
            'next': char_list[index+1] if index < len(char_list)-1 else None,
            'position': 'start' if index == 0 else ('end' if index == len(char_list)-1 else 'middle')
        }


# Instancia global
post_corrector = OCRPostCorrector()