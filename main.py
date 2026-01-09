"""
Sistema OCR Principal - Examen Final IA
Autor: [Tu Nombre]
Fecha: 2024-12-15

Pipeline completo de OCR sin librerías pre-entrenadas.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Importar módulos propios
from src.utils.config import load_config
from src.preprocessing.pipeline import PreprocessingPipeline
from segmentation.text_segmenter import TextSegmenter
from src.recognition.digital_ocr import DigitalOCR
from src.postprocessing.formatter import OutputFormatter


class OCRSystem:
    """
    Sistema OCR completo de extremo a extremo.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Inicializa el sistema OCR.
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        print("🚀 Inicializando Sistema OCR...")
        
        # Cargar configuración
        self.config = load_config(config_path)
        
        # Inicializar componentes
        self.preprocessor = PreprocessingPipeline(self.config)
        self.segmenter = TextSegmenter(self.config)
        self.recognizer = DigitalOCR(self.config)
        self.formatter = OutputFormatter(self.config)
        
        print("✅ Sistema inicializado correctamente\n")
    
    def process_image(self, image_path, output_dir=None):
        """
        Procesa una imagen completa y genera el texto reconocido.
        
        Args:
            image_path (str): Ruta a la imagen de entrada
            output_dir (str): Directorio de salida (opcional)
            
        Returns:
            dict: Resultados del procesamiento
        """
        print(f"📄 Procesando: {image_path}")
        
        # 1. CARGA DE IMAGEN
        print("  └─ Cargando imagen...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # 2. PREPROCESAMIENTO
        print("  └─ Preprocesando imagen...")
        processed = self.preprocessor.process(image)
        
        # Guardar imagen preprocesada para debug
        debug_dir = Path(self.config['paths']['output']) / 'debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / 'preprocessed.png'), processed)
        
        # 3. SEGMENTACIÓN
        print("  └─ Segmentando texto...")
        segments = self.segmenter.segment(processed)
        
        print(f"      • {len(segments['lines'])} líneas detectadas")
        print(f"      • {len(segments['words'])} palabras detectadas")
        print(f"      • {len(segments['characters'])} caracteres detectados")
        
        # Guardar visualización de segmentos
        self._visualize_segments(image, segments, debug_dir)
        
        # 4. RECONOCIMIENTO
        print("  └─ Reconociendo caracteres...")
        recognized_text = self.recognizer.recognize(segments)
        
        # 5. FORMATEO Y GUARDADO
        print("  └─ Guardando resultados...")
        if output_dir is None:
            output_dir = self.config['paths']['output']
        
        output_path = self.formatter.save(
            recognized_text,
            image_path,
            segments
        )
        
        results = {
            'input_path': image_path,
            'output_path': output_path,
            'text': recognized_text,
            'stats': {
                'lines': len(segments['lines']),
                'words': len(segments['words']),
                'characters': len(segments['characters'])
            }
        }
        
        print(f"✅ Procesamiento completado")
        print(f"   Salida: {output_path}\n")
        
        return results
    
    def _visualize_segments(self, original_image, segments, output_dir):
        """
        Crea visualización de la segmentación.
        """
        vis_image = original_image.copy()
        
        # Dibujar líneas
        for line in segments['lines']:
            # Aquí deberías tener las coordenadas del bounding box
            # Por simplicidad, esto es un placeholder
            pass
        
        cv2.imwrite(str(output_dir / 'segmentation.png'), vis_image)
    
    def batch_process(self, input_dir, output_dir=None):
        """
        Procesa todas las imágenes en un directorio.
        
        Args:
            input_dir (str): Directorio con imágenes
            output_dir (str): Directorio de salida
            
        Returns:
            list: Lista de resultados
        """
        input_path = Path(input_dir)
        allowed_formats = self.config['input']['allowed_formats']
        
        image_files = []
        for ext in allowed_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
        
        print(f"📁 Procesando {len(image_files)} imágenes...\n")
        
        results = []
        for image_file in image_files:
            try:
                result = self.process_image(str(image_file), output_dir)
                results.append(result)
            except Exception as e:
                print(f"❌ Error procesando {image_file}: {str(e)}")
                continue
        
        print(f"\n✅ Procesamiento por lotes completado")
        print(f"   {len(results)}/{len(image_files)} imágenes procesadas exitosamente")
        
        return results


def main():
    """
    Función principal del programa.
    """
    print("=" * 60)
    print("   SISTEMA OCR - EXAMEN FINAL INTELIGENCIA ARTIFICIAL")
    print("   Universidad: [Tu Universidad]")
    print("   Asignatura: 051 - Inteligencia Artificial")
    print("=" * 60 + "\n")
    
    # Inicializar sistema
    ocr = OCRSystem(config_path="config.yaml")
    
    # Ejemplo de uso
    if len(sys.argv) > 1:
        # Procesar imagen desde argumentos
        image_path = sys.argv[1]
        ocr.process_image(image_path)
    else:
        # Procesar directorio de prueba
        test_dir = "data/raw/test_samples"
        if os.path.exists(test_dir):
            ocr.batch_process(test_dir)
        else:
            print("⚠️  No se encontraron imágenes de prueba")
            print("   Coloca imágenes en: data/raw/test_samples/")
            print("\n💡 Uso: python main.py <ruta_imagen>")


if __name__ == "__main__":
    main()