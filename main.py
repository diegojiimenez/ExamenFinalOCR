"""
Sistema OCR Mejorado - Reconocimiento Universal de Texto
"""

import sys
import os

def main():
    """
    Punto de entrada principal con soporte CLI y GUI
    """
    
    # Modo CLI
    if len(sys.argv) > 1:
        from core.predictor import predict_cli
        command = sys.argv[1]
        
        if command == "predict" and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            predict_cli(image_path)
        elif command == "test":
            # Procesar todas las imágenes de prueba
            from pathlib import Path
            test_dir = Path("data/test_images")
            if test_dir.exists():
                for img in test_dir.glob("*.png"):
                    predict_cli(str(img))
        else:
            print("Uso:")
            print("  python main.py predict <imagen>  - Predecir una imagen")
            print("  python main.py test              - Probar todas las imágenes")
            print("  python main.py                   - Abrir interfaz gráfica")
    else:
        # Modo GUI
        from gui.modern_interface import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()