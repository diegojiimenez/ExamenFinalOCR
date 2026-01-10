"""
Cargador de datos para entrenamiento y pruebas
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
from utils.config import config


class DataLoader:
    """
    Carga y organiza datos de las diferentes carpetas
    """
    
    def __init__(self):
        self.config = config
    
    def get_dataset_images(self, dataset_name='all'):
        """
        Obtiene imágenes de los datasets con manejo correcto de rutas UTF-8
        
        Args:
            dataset_name: 'dataset_ia', 'dataset_completo', 'dataset_completo2' o 'all'
        
        Returns:
            Lista de rutas de imágenes
        """
        if dataset_name == 'all':
            datasets = self.config.get_all_dataset_paths()
            all_images = []
            for name, path in datasets.items():
                if path and path.exists():
                    images = self._scan_directory(path)
                    print(f"  📁 {name}: {len(images)} imágenes")
                    all_images.extend(images)
            return all_images
        else:
            path = self.config.get_path(dataset_name)
            if path and path.exists():
                return self._scan_directory(path)
            return []
    
    def get_referencia_images(self, categoria=None):
        """
        Obtiene imágenes de referencia
        
        Args:
            categoria: 'mayusculas', 'minusculas', 'numeros' o None (todas)
        
        Returns:
            Dict con categorías y sus imágenes
        """
        ref_paths = self.config.get_referencia_paths()
        
        if categoria:
            path = ref_paths.get(categoria)
            if path and path.exists():
                return {categoria: self._scan_directory(path)}
            return {}
        
        # Todas las categorías
        result = {}
        for cat, path in ref_paths.items():
            if path and path.exists():
                result[cat] = self._scan_directory(path)
        
        return result
    
    def get_pruebas_images(self):
        """Obtiene todas las imágenes de prueba"""
        pruebas_path = self.config.get_path('pruebas')
        if pruebas_path and pruebas_path.exists():
            return self._scan_directory(pruebas_path)
        return []
    
    def _scan_directory(self, directory):
        """
        Escanea un directorio recursivamente buscando imágenes
        CON MANEJO CORRECTO DE UTF-8
        
        Args:
            directory: Path del directorio
        
        Returns:
            Lista de rutas de imágenes
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        images = []
        
        try:
            # Usar os.walk con encoding correcto
            for root, dirs, files in os.walk(str(directory)):
                for file in files:
                    try:
                        # Construir ruta completa con manejo de encoding
                        file_path = Path(root) / file
                        
                        # Verificar extensión
                        if file_path.suffix.lower() in image_extensions:
                            # Verificar que el archivo existe y es accesible
                            if file_path.exists() and os.access(str(file_path), os.R_OK):
                                images.append(file_path)
                    except (UnicodeDecodeError, OSError) as e:
                        # Saltar archivos con problemas de encoding
                        continue
        except Exception as e:
            print(f"⚠️  Error al escanear directorio {directory}: {e}")
        
        return sorted(images)
    
    def load_image(self, image_path, mode='L'):
        """
        Carga una imagen con manejo de errores UTF-8
        
        Args:
            image_path: Ruta de la imagen
            mode: Modo de PIL ('L' para escala de grises, 'RGB' para color)
        
        Returns:
            Imagen de PIL o None si falla
        """
        try:
            # Usar Path para manejo correcto de rutas
            path = Path(image_path)
            if not path.exists():
                return None
            
            return Image.open(str(path)).convert(mode)
        except Exception as e:
            print(f"⚠️  Error al cargar {image_path}: {e}")
            return None
    
    def get_dataset_stats(self):
        """Obtiene estadísticas de los datasets"""
        stats = {
            'dataset': {},
            'referencia': {},
            'pruebas': 0
        }
        
        # Datasets
        for name in ['dataset_ia', 'dataset_completo', 'dataset_completo2']:
            images = self.get_dataset_images(name)
            stats['dataset'][name] = len(images)
        
        # Referencia
        ref_images = self.get_referencia_images()
        for cat, images in ref_images.items():
            stats['referencia'][cat] = len(images)
        
        # Pruebas
        stats['pruebas'] = len(self.get_pruebas_images())
        
        return stats
    
    def print_stats(self):
        """Imprime estadísticas de los datos"""
        stats = self.get_dataset_stats()
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS DE DATOS")
        print("="*60)
        
        print("\n📦 DATASETS DE ENTRENAMIENTO:")
        for name, count in stats['dataset'].items():
            print(f"   {name}: {count} imágenes")
        
        print("\n📚 REFERENCIA:")
        for cat, count in stats['referencia'].items():
            print(f"   {cat}: {count} imágenes")
        
        print(f"\n🧪 PRUEBAS: {stats['pruebas']} imágenes")
        print("="*60 + "\n")


# Instancia global
data_loader = DataLoader()