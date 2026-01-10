"""
Sistema OCR - Compatible con proyecto Examen Final
"""

import sys
from pathlib import Path


def main():
    """Punto de entrada"""
    
    if len(sys.argv) > 1:
        from core.predictor import predict_cli, predict_with_visualization
        from utils.data_loader import data_loader
        from utils.config import config
        
        command = sys.argv[1]
        
        if command == "predict" and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            visualize = "--viz" in sys.argv
            predict_cli(image_path, visualize=visualize)
        
        elif command == "visualize" and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            predict_with_visualization(image_path)
        
        elif command == "train":
            from core.model_training import train_model
            train_model()
        
        elif command == "test":
            pruebas = data_loader.get_pruebas_images()
            if not pruebas:
                print("⚠️  No hay imágenes de prueba")
                return
            
            for img in pruebas[:5]:  # Primeras 5
                predict_cli(str(img), visualize=False)
        
        elif command == "stats":
            data_loader.print_stats()
        
        elif command == "verify":
            config.verify_paths()
        
        else:
            print("Uso:")
            print("=" * 60)
            print("  python main.py predict <img>        - Predecir")
            print("  python main.py predict <img> --viz  - Predecir + visualizar")
            print("  python main.py visualize <img>      - Solo visualizar")
            print("  python main.py train                - Entrenar")
            print("  python main.py test                 - Probar dataset")
            print("  python main.py stats                - Estadísticas")
            print("  python main.py                      - Abrir GUI")
            print("=" * 60)
    else:
        from gui.modern_interface import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()