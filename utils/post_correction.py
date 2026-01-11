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
        
        # PASO 0: Eliminar puntos huérfanos
        corrected_chars, removed_dots = self._remove_orphan_dots(
            corrected_chars, confidences, char_images, debug
        )
        
        if len(removed_dots) > 0:
            corrections_made.append(f"Puntos huérfanos eliminados: {len(removed_dots)}")
            confidences = [conf for i, conf in enumerate(confidences) if i not in removed_dots]
        
        # Determinar modo
        uppercase_count = sum(1 for c in corrected_chars if c.isupper())
        lowercase_count = sum(1 for c in corrected_chars if c.islower())
        total_letters = uppercase_count + lowercase_count
        
        if total_letters == 0:
            if debug:
                print(f"   Modo: SIN LETRAS (solo números/símbolos) - No aplicar corrección")
            return ''.join(corrected_chars)
        
        is_uppercase_mode = uppercase_count > lowercase_count
        
        if debug:
            print(f"   Modo detectado: {'MAYÚSCULAS' if is_uppercase_mode else 'minúsculas'}")
            print(f"   (May: {uppercase_count}, Min: {lowercase_count})")
        
        # Analizar cada carácter
        for i, (char, conf) in enumerate(zip(corrected_chars, confidences)):
            
            if char in ['P', 'p']:
                if conf < 0.50:
                    corrected = self._analyze_thin_character(
                        char, i, corrected_chars, char_images, 
                        confidences[i], is_uppercase_mode, debug
                    )
                    if corrected != char:
                        corrected_chars[i] = corrected
                        corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char == 'f':
                # 🔧 MEJORADO: Analizar 'f' si conf < 95% Y verificar geometría
                if conf < 0.95:  # Subido de 0.85 a 0.95
                    # Verificar geometría primero
                    should_analyze = self._should_analyze_f_as_i(
                        char, i, corrected_chars, char_images, conf, debug
                    )
                    
                    if should_analyze:
                        corrected = self._analyze_thin_character(
                            char, i, corrected_chars, char_images, 
                            confidences[i], is_uppercase_mode, debug
                        )
                        if corrected != char:
                            corrected_chars[i] = corrected
                            corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['l', '1', '|']:
                if conf < 0.40:
                    if self._should_be_letter(char, i, corrected_chars, debug):
                        corrected = self._analyze_thin_character(
                            char, i, corrected_chars, char_images, 
                            confidences[i], is_uppercase_mode, debug
                        )
                        if corrected != char:
                            corrected_chars[i] = corrected
                            corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['I', 'i'] and conf < 0.85:
                corrected = self._validate_i_character(
                    char, i, corrected_chars, is_uppercase_mode, debug
                )
                if corrected != char:
                    corrected_chars[i] = corrected
                    corrections_made.append(f"{i}: '{char}'→'{corrected}' ({conf:.1%})")
            
            elif char in ['O', '0', 'o'] and conf < 0.70:
                corrected = self._analyze_o_vs_zero(char, i, corrected_chars, debug)
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
    
    def _remove_orphan_dots(self, char_list, confidences, char_images, debug):
        """
        🆕 Elimina puntos huérfanos que son parte de 'i'/'j' mal segmentadas.
        
        Detecta:
        - Caracteres predichos como '.' o símbolos pequeños
        - Con área muy pequeña
        - Que podrían ser el punto de 'i'/'j'
        
        Returns:
            (char_list_limpio, set_de_indices_eliminados)
        """
        if not char_images or len(char_list) != len(char_images):
            return char_list, set()
        
        indices_to_remove = set()
        
        for i, (char, conf) in enumerate(zip(char_list, confidences)):
            # Detectar posibles puntos
            if char in ['.', ',', "'", '`', '-', '_'] or (not char.isalnum() and conf < 0.70):
                # Analizar geometría
                img = char_images[i]
                h, w = img.shape[:2] if len(img.shape) > 1 else (img.shape[0], 1)
                area = h * w
                aspect_ratio = w / h if h > 0 else 0
                
                # ¿Es muy pequeño y casi cuadrado? (posible punto de 'i'/'j')
                is_likely_dot = area < 200 and 0.5 < aspect_ratio < 2.0
                
                if is_likely_dot:
                    indices_to_remove.add(i)
                    if debug:
                        print(f"      🗑️ Eliminando punto huérfano [{i}]: '{char}' (área={area}, AR={aspect_ratio:.2f})")
        
        # Eliminar caracteres
        cleaned_chars = [char for i, char in enumerate(char_list) if i not in indices_to_remove]
        
        return cleaned_chars, indices_to_remove
    
    def _should_be_letter(self, char, index, char_list, debug):
        """
        Verifica si un carácter '1'/'l'/'|' debería ser letra o número
        """
        context = self._get_context(char_list, index)
        
        prev_is_letter = context['prev'] and context['prev'].isalpha()
        next_is_letter = context['next'] and context['next'].isalpha()
        
        prev_is_digit = context['prev'] and context['prev'].isdigit()
        next_is_digit = context['next'] and context['next'].isdigit()
        
        if prev_is_letter or next_is_letter:
            if debug:
                print(f"         → Cerca de letras, podría ser 'I'/'i'")
            return True
        
        if prev_is_digit or next_is_digit:
            if debug:
                print(f"         → Cerca de números, es '1'")
            return False
        
        if debug:
            print(f"         → Sin contexto claro, mantener como número")
        return False
    
    def _analyze_thin_character(self, char, index, char_list, char_images, 
                                confidence, is_uppercase_mode, debug):
        """
        Analiza UN carácter delgado que podría ser 'I' o 'i'
        """
        if char_images and index < len(char_images):
            img = char_images[index]
            h, w = img.shape[:2] if len(img.shape) > 1 else (img.shape[0], 1)
            aspect_ratio = w / h if h > 0 else 0
            
            if debug:
                print(f"      [{index}] '{char}' | AR: {aspect_ratio:.2f} | Conf: {confidence:.1%}")
            
            if aspect_ratio < 0.35:
                result = 'I' if is_uppercase_mode else 'i'
                if debug:
                    print(f"         → AR < 0.35, modo {('MAY' if is_uppercase_mode else 'min')} → '{result}'")
                return result
            
            elif aspect_ratio < 0.50 and confidence < 0.50:
                result = 'I' if is_uppercase_mode else 'i'
                if debug:
                    print(f"         → AR ambiguo + baja conf → '{result}'")
                return result
        
        context = self._get_context(char_list, index)
        
        if context['prev'] and context['prev'].isupper():
            if debug:
                print(f"         → Después de mayúscula → 'I'")
            return 'I'
        
        if context['next'] and context['next'].isupper():
            if debug:
                print(f"         → Antes de mayúscula → 'I'")
            return 'I'
        
        if context['prev'] and context['prev'].islower():
            if debug:
                print(f"         → Después de minúscula → 'i'")
            return 'i'
        
        if context['next'] and context['next'].islower():
            if debug:
                print(f"         → Antes de minúscula → 'i'")
            return 'i'
        
        return 'I' if is_uppercase_mode else 'i'
    
    def _validate_i_character(self, char, index, char_list, is_uppercase_mode, debug):
        """
        Valida si 'I' o 'i' reconocida es correcta
        """
        context = self._get_context(char_list, index)
        
        if char == 'I':
            if context['prev'] and context['prev'].islower() and context['next'] and context['next'].islower():
                if debug:
                    print(f"      [{index}] 'I' entre minúsculas → 'i'")
                return 'i'
        
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
    
    def _should_analyze_f_as_i(self, char, index, char_list, char_images, confidence, debug):
        """
        🆕 Decide si una 'f' detectada debería analizarse como posible 'i'.
        
        Reglas:
        1. Si aspect ratio < 0.40 → MUY delgada, probablemente 'i' fusionada
        2. Si confianza < 85% Y está entre minúsculas → probablemente 'i'
        3. Si confianza < 70% → siempre analizar
        
        Returns:
            True si debe analizarse, False si dejar como 'f'
        """
        # Regla 1: Siempre analizar si confianza < 70%
        if confidence < 0.70:
            if debug:
                print(f"      [{index}] 'f' con conf < 70% → analizar")
            return True
        
        # Regla 2: Analizar si geometría indica 'i' (muy delgada)
        if char_images and index < len(char_images):
            img = char_images[index]
            h, w = img.shape[:2] if len(img.shape) > 1 else (img.shape[0], 1)
            aspect_ratio = w / h if h > 0 else 0
            
            if debug:
                print(f"      [{index}] 'f' ({confidence:.1%}) | AR: {aspect_ratio:.2f}")
            
            # 'i' fusionada tiende a ser delgada (aspect < 0.50)
            # 'f' real es más ancha (aspect > 0.50)
            if aspect_ratio < 0.50:
                if debug:
                    print(f"         → AR < 0.50, probablemente 'i' fusionada → analizar")
                return True
        
        # Regla 3: Si está entre minúsculas Y confianza < 90%
        context = self._get_context(char_list, index)
        prev_is_lower = context['prev'] and context['prev'].islower()
        next_is_lower = context['next'] and context['next'].islower()
        
        if prev_is_lower and next_is_lower and confidence < 0.90:
            if debug:
                print(f"         → Entre minúsculas con conf < 90% → analizar")
            return True
        
        if debug:
            print(f"         → No cumple criterios → mantener como 'f'")
        return False


# Instancia global
post_corrector = OCRPostCorrector()