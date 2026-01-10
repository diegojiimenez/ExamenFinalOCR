"""
Sistema OCR Mejorado - Reconocimiento Universal de Texto
"""

import sys
import os
from pathlib import Path


def main():
    """
    Punto de entrada principal con soporte CLI y GUI
    """
    
    # Modo CLI
    if len(sys.argv) > 1:
        from core.predictor import predict_cli
        from utils.data_loader import data_loader
        from utils.config import config
        
        command = sys.argv[1]
        
        if command == "train":
            # 🆕 NUEVO: Entrenar modelo
            from core.model_training import train_model
            print("🚀 Iniciando entrenamiento del modelo...")
            train_model()
            
        elif command == "predict" and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            predict_cli(image_path)
            
        elif command == "test":
            # Procesar imágenes de prueba
            print("🧪 Ejecutando pruebas...")
            pruebas = data_loader.get_pruebas_images()
            
            if not pruebas:
                print("⚠️  No hay imágenes de prueba en data/pruebas/")
                return
            
            print(f"📊 Encontradas {len(pruebas)} imágenes de prueba\n")
            
            for img in pruebas:
                print(f"\n{'─'*60}")
                predict_cli(str(img))
                
        elif command == "stats":
            # Mostrar estadísticas de datos
            data_loader.print_stats()
            
        elif command == "verify":
            # Verificar rutas de configuración
            print("🔍 Verificando rutas de configuración...")
            config.verify_paths()
            
        elif command == "referencia":
            # Probar con imágenes de referencia
            categoria = sys.argv[2] if len(sys.argv) >= 3 else None
            ref_images = data_loader.get_referencia_images(categoria)
            
            if not ref_images:
                print(f"⚠️  No se encontraron imágenes de referencia")
                return
            
            for cat, images in ref_images.items():
                print(f"\n{'='*60}")
                print(f"📚 Categoría: {cat.upper()}")
                print('='*60)
                
                for img in images[:5]:  # Primeras 5 de cada categoría
                    predict_cli(str(img))
        
        else:
            print("Uso del Sistema OCR:")
            print("=" * 60)
            print("  python main.py train                - 🆕 Entrenar modelo")
            print("  python main.py predict <imagen>     - Predecir una imagen")
            print("  python main.py test                 - Probar imágenes de prueba")
            print("  python main.py referencia [cat]     - Probar imágenes de referencia")
            print("  python main.py stats                - Ver estadísticas de datos")
            print("  python main.py verify               - Verificar configuración")
            print("  python main.py                      - Abrir interfaz gráfica")
            print("=" * 60)
    else:
        # Modo GUI
        from gui.modern_interface import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()