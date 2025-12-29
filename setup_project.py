"""
Script de configuración inicial del proyecto OCR

Este script crea toda la estructura de carpetas y archivos iniciales
necesarios para comenzar el desarrollo del proyecto.

Uso:
    python setup_project.py
"""

import os
import sys


def create_directory_structure():
    """
    Crea toda la estructura de directorios del proyecto.
    """
    directories = [
        # Datos
        "data/raw/digital_text",
        "data/raw/handwritten_text",
        "data/raw/test_samples",
        "data/datasets/mnist",
        "data/datasets/emnist",
        "data/datasets/custom_handwriting",
        "data/datasets/fonts",
        "data/processed",
        
        # Modelos
        "models/checkpoints",
        
        # Salidas
        "output/text",
        "output/images",
        "output/tables",
        "output/barcodes_qr",
        
        # Código fuente
        "src/preprocessing",
        "src/feature_extraction",
        "src/recognition",
        "src/optional_features",
        "src/training",
        "src/postprocessing",
        "src/utils",
        
        # Notebooks
        "notebooks",
        
        # Tests
        "tests",
        
        # Documentación
        "docs",
        
        # Logs
        "logs",
        
        # Scripts auxiliares
        "scripts"
    ]
    
    print("Creando estructura de directorios...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("\n✅ Estructura de directorios creada exitosamente!\n")


def create_init_files():
    """
    Crea archivos __init__.py en todos los paquetes de Python.
    """
    packages = [
        "src",
        "src/preprocessing",
        "src/feature_extraction",
        "src/recognition",
        "src/optional_features",
        "src/training",
        "src/postprocessing",
        "src/utils",
        "tests"
    ]
    
    print("Creando archivos __init__.py...")
    for package in packages:
        init_file = os.path.join(package, "__init__.py")
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(f'"""\n{package.split("/")[-1]} package\n"""\n')
        print(f"  ✓ {init_file}")
    
    print("\n✅ Archivos __init__.py creados!\n")


def create_placeholder_modules():
    """
    Crea módulos de Python con estructura básica como plantillas.
    """
    modules = {
        "src/utils/config.py": '''"""
Módulo de gestión de configuración
"""

import yaml


def load_config(config_path="config.yaml"):
    """
    Carga la configuración desde un archivo YAML.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        dict: Configuración cargada
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
''',
        
        "src/preprocessing/image_loader.py": '''"""
Módulo de carga de imágenes
"""

import cv2
import numpy as np
from PIL import Image


class ImageLoader:
    """
    Clase para cargar y validar imágenes de entrada.
    """
    
    def __init__(self, config):
        """
        Inicializa el cargador de imágenes.
        
        Args:
            config (dict): Configuración del sistema
        """
        self.config = config
        self.allowed_formats = config['input']['allowed_formats']
    
    def load(self, image_path):
        """
        Carga una imagen desde una ruta.
        
        Args:
            image_path (str): Ruta a la imagen
            
        Returns:
            numpy.ndarray: Imagen cargada
            
        Raises:
            ValueError: Si el formato no es válido
        """
        # TODO: Implementar carga de imagen
        # Verificar formato
        # Cargar con OpenCV o PIL
        # Validar dimensiones
        pass
''',
        
        "src/preprocessing/noise_reduction.py": '''"""
Módulo de reducción de ruido en imágenes
"""

import cv2
import numpy as np


class NoiseReducer:
    """
    Clase para aplicar técnicas de reducción de ruido.
    """
    
    def __init__(self, config):
        self.config = config
        self.method = config['preprocessing']['noise_reduction']['method']
        self.kernel_size = config['preprocessing']['noise_reduction']['kernel_size']
    
    def reduce(self, image):
        """
        Aplica reducción de ruido a una imagen.
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            numpy.ndarray: Imagen sin ruido
        """
        # TODO: Implementar métodos de reducción de ruido
        # - Filtro Gaussiano
        # - Filtro Bilateral
        # - Filtro de Mediana
        pass
''',
        
        "src/preprocessing/binarization.py": '''"""
Módulo de binarización de imágenes
"""

import cv2
import numpy as np


class Binarizer:
    """
    Clase para binarizar imágenes (convertir a blanco y negro).
    """
    
    def __init__(self, config):
        self.config = config
        self.method = config['preprocessing']['binarization']['method']
    
    def binarize(self, image):
        """
        Binariza una imagen usando el método configurado.
        
        Args:
            image (numpy.ndarray): Imagen en escala de grises
            
        Returns:
            numpy.ndarray: Imagen binarizada
        """
        # TODO: Implementar métodos de binarización
        # - Método de Otsu
        # - Binarización adaptativa
        # - Método de Sauvola
        pass
''',
        
        "src/preprocessing/deskew.py": '''"""
Módulo de corrección de inclinación
"""

import cv2
import numpy as np


class Deskewer:
    """
    Clase para corregir la inclinación de imágenes.
    """
    
    def __init__(self, config):
        self.config = config
        self.max_angle = config['preprocessing']['deskew']['max_angle']
    
    def deskew(self, image):
        """
        Corrige la inclinación de una imagen.
        
        Args:
            image (numpy.ndarray): Imagen inclinada
            
        Returns:
            numpy.ndarray: Imagen corregida
        """
        # TODO: Implementar corrección de inclinación
        # - Detectar ángulo de inclinación
        # - Aplicar rotación
        pass
''',
        
        "src/preprocessing/segmentation.py": '''"""
Módulo de segmentación de texto
"""

import cv2
import numpy as np


class Segmenter:
    """
    Clase para segmentar texto en líneas, palabras y caracteres.
    """
    
    def __init__(self, config):
        self.config = config
    
    def segment(self, image):
        """
        Segmenta una imagen en líneas, palabras y caracteres.
        
        Args:
            image (numpy.ndarray): Imagen binarizada
            
        Returns:
            dict: Diccionario con segmentos
                {
                    'lines': list,
                    'words': list,
                    'characters': list
                }
        """
        # TODO: Implementar segmentación
        # - Proyección horizontal para líneas
        # - Proyección vertical para palabras
        # - Contornos para caracteres
        pass
''',
        
        "src/recognition/digital_ocr.py": '''"""
Módulo de reconocimiento de texto impreso
"""

import numpy as np


class DigitalOCR:
    """
    Clase para reconocer texto impreso (tipografía digital).
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None  # TODO: Cargar modelo entrenado
    
    def recognize(self, segments):
        """
        Reconoce texto en los segmentos proporcionados.
        
        Args:
            segments (dict): Segmentos de caracteres
            
        Returns:
            str: Texto reconocido
        """
        # TODO: Implementar reconocimiento
        # - Preprocesar caracteres
        # - Aplicar modelo CNN
        # - Reconstruir texto
        pass
''',
        
        "src/recognition/handwritten_ocr.py": '''"""
Módulo de reconocimiento de texto manuscrito
"""

import numpy as np


class HandwrittenOCR:
    """
    Clase para reconocer texto manuscrito.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None  # TODO: Cargar modelo entrenado
    
    def recognize(self, segments):
        """
        Reconoce texto manuscrito en los segmentos.
        
        Args:
            segments (dict): Segmentos de caracteres
            
        Returns:
            str: Texto reconocido
        """
        # TODO: Implementar reconocimiento
        # - Preprocesar escritura manual
        # - Aplicar modelo CNN/LSTM
        # - Reconstruir texto
        pass
''',
        
        "src/postprocessing/formatter.py": '''"""
Módulo de formateo de salida
"""

import os
from datetime import datetime


class OutputFormatter:
    """
    Clase para formatear y guardar resultados del OCR.
    """
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config['paths']['output']
    
    def save(self, text, image_path, segments=None):
        """
        Guarda el texto reconocido en un archivo.
        
        Args:
            text (str): Texto reconocido
            image_path (str): Ruta de la imagen original
            segments (dict): Segmentos detectados (opcional)
            
        Returns:
            str: Ruta del archivo de salida
        """
        # TODO: Implementar guardado de resultados
        # - Crear nombre de archivo
        # - Guardar en formato .txt
        # - Opcionalmente guardar metadata en JSON
        pass
'''
    }
    
    print("Creando módulos de plantilla...")
    for filepath, content in modules.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ {filepath}")
    
    print("\n✅ Módulos de plantilla creados!\n")


def create_gitignore():
    """
    Crea archivo .gitignore apropiado para el proyecto.
    """
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Modelos entrenados (pueden ser grandes)
models/*.h5
models/*.pkl
models/*.pt
models/*.pth

# Datos (evitar subir imágenes grandes)
data/raw/
data/processed/
!data/raw/.gitkeep
!data/processed/.gitkeep

# Outputs
output/
logs/
*.log

# Sistema operativo
.DS_Store
Thumbs.db

# Archivos temporales
*.tmp
*.bak
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ Archivo .gitignore creado!\n")


def create_empty_markers():
    """
    Crea archivos .gitkeep en carpetas que deben estar en git pero vacías.
    """
    marker_dirs = [
        "data/raw",
        "data/processed",
        "output",
        "logs"
    ]
    
    for directory in marker_dirs:
        gitkeep = os.path.join(directory, ".gitkeep")
        with open(gitkeep, 'w') as f:
            pass


def print_next_steps():
    """
    Imprime los siguientes pasos a seguir.
    """
    print("\n" + "="*60)
    print("🎉 ¡PROYECTO CONFIGURADO EXITOSAMENTE!")
    print("="*60)
    print("\n📋 PRÓXIMOS PASOS:\n")
    print("1. Instalar dependencias:")
    print("   pip install -r requirements.txt\n")
    print("2. Revisar y ajustar config.yaml según tus necesidades\n")
    print("3. Colocar tus imágenes de prueba en data/raw/\n")
    print("4. Descargar datasets (MNIST, EMNIST) para entrenamiento\n")
    print("5. Comenzar a implementar los módulos marcados con TODO\n")
    print("6. Sugerencia de orden de implementación:")
    print("   - Preprocesamiento (image_loader, binarization)")
    print("   - Segmentación")
    print("   - Modelo simple de reconocimiento")
    print("   - Entrenamiento y evaluación")
    print("   - Funcionalidades opcionales\n")
    print("7. Mantener registro de experimentos en logs/\n")
    print("8. Documentar progreso en docs/\n")
    print("="*60)
    print("\n💡 TIP: Usa notebooks/ para experimentación inicial")
    print("💡 TIP: Revisa el README.md para más información\n")


def main():
    """
    Función principal que ejecuta toda la configuración.
    """
    print("\n" + "="*60)
    print("    CONFIGURACIÓN INICIAL DEL PROYECTO OCR")
    print("    Asignatura: 051 - Inteligencia Artificial")
    print("="*60 + "\n")
    
    try:
        create_directory_structure()
        create_init_files()
        create_placeholder_modules()
        create_gitignore()
        create_empty_markers()
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Error durante la configuración: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
