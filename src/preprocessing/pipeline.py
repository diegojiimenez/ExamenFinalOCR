"""
Pipeline completo de preprocesamiento de imágenes
"""

import cv2
import numpy as np


class PreprocessingPipeline:
    """
    Pipeline de preprocesamiento para preparar imágenes para OCR.
    """
    
    def __init__(self, config):
        self.config = config
    
    def process(self, image):
        """
        Aplica todo el preprocesamiento a una imagen.
        
        Args:
            image (numpy.ndarray): Imagen original
            
        Returns:
            numpy.ndarray: Imagen procesada
        """
        # 1. Convertir a escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 2. Reducción de ruido - Filtro bilateral mantiene bordes
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 3. Normalizar contraste con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 4. Binarización adaptativa (mejor para iluminación no uniforme)
        binary = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # 5. Invertir si el fondo es negro (texto blanco)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # 6. Corrección de inclinación (deskew)
        binary = self._deskew(binary)
        
        # 7. Morfología para limpiar ruido pequeño
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _deskew(self, image):
        """
        Corrige la inclinación de la imagen.
        """
        # Detectar coordenadas de píxeles negros
        coords = np.column_stack(np.where(image == 0))
        
        if len(coords) == 0:
            return image
        
        # Calcular ángulo de inclinación
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Limitar ángulo máximo
        if abs(angle) > self.config['preprocessing']['deskew']['max_angle']:
            return image
        
        # Aplicar rotación
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated