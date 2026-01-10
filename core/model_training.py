"""
Entrenamiento del modelo OCR con arquitectura CNN mejorada
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
from utils.config import config
from utils.data_loader import data_loader


class OCRModelTrainer:
    """
    Entrenador del modelo OCR
    """
    
    def __init__(self, img_size=28):
        self.img_size = img_size
        self.char_map = self._build_char_map()
        self.num_classes = len(self.char_map)
        self.model = None
    
    def _build_char_map(self):
        """Construye el mapa de caracteres desde configuración"""
        char_map_str = config.get('model.char_map', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return {i: char for i, char in enumerate(char_map_str)}
    
    def load_dataset(self):
        """
        Carga el dataset desde las carpetas configuradas
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "="*60)
        print("📦 CARGANDO DATASET")
        print("="*60)
        
        # Obtener todas las imágenes de dataset
        all_images = data_loader.get_dataset_images('all')
        
        if not all_images:
            raise ValueError("⚠️  No se encontraron imágenes en los datasets")
        
        print(f"📊 Total de imágenes encontradas: {len(all_images)}")
        
        X = []
        y = []
        
        # Procesar cada imagen
        for img_path in all_images:
            # Extraer etiqueta del nombre del archivo o estructura
            label = self._extract_label(img_path)
            
            if label is None:
                continue
            
            # Cargar y procesar imagen
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Redimensionar a 28x28
            img_resized = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalizar
            img_normalized = img_resized.astype('float32') / 255.0
            
            X.append(img_normalized)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape para CNN
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        print(f"✅ Datos cargados: {len(X)} muestras")
        print(f"📊 Distribución de clases: {np.unique(y, return_counts=True)}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"🔹 Train: {len(X_train)} muestras")
        print(f"🔹 Test: {len(X_test)} muestras")
        print("="*60 + "\n")
        
        return X_train, X_test, y_train, y_test
    
    def _extract_label(self, img_path):
        """
        Extrae la etiqueta (clase) de la ruta de la imagen
        
        Asume estructura: .../dataset/LETRA/imagen.png
        O nombre de archivo: A_001.png, B_002.png, etc.
        """
        img_path = Path(img_path)
        
        # Método 1: Carpeta padre es la etiqueta
        parent_name = img_path.parent.name
        if len(parent_name) == 1 and parent_name in self.char_map.values():
            return list(self.char_map.keys())[list(self.char_map.values()).index(parent_name)]
        
        # Método 2: Primer carácter del nombre del archivo
        filename = img_path.stem
        if filename and filename[0] in self.char_map.values():
            return list(self.char_map.keys())[list(self.char_map.values()).index(filename[0])]
        
        return None
    
    def build_model(self):
        """
        Construye la arquitectura CNN mejorada
        """
        print("🏗️  Construyendo arquitectura del modelo...")
        
        model = Sequential([
            # Primera capa convolucional
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Segunda capa convolucional
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Tercera capa convolucional
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten y capas densas
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            # Capa de salida
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ Modelo construido exitosamente")
        print(f"📊 Total de parámetros: {model.count_params():,}")
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """
        Entrena el modelo
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("="*60)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/trained_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        
        # Evaluación final
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Precisión en test: {test_acc:.4f}")
        print(f"📉 Pérdida en test: {test_loss:.4f}")
        print("="*60 + "\n")
        
        return history
    
    def save_model(self, filepath='models/trained_model.h5'):
        """Guarda el modelo entrenado"""
        os.makedirs('models', exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Modelo guardado en: {filepath}")


def train_model():
    """
    Función principal para entrenar el modelo
    """
    print("\n" + "🔤"*30)
    print("SISTEMA DE ENTRENAMIENTO OCR")
    print("🔤"*30 + "\n")
    
    # Crear entrenador
    trainer = OCRModelTrainer()
    
    # Cargar datos
    X_train, X_test, y_train, y_test = trainer.load_dataset()
    
    # Construir modelo
    trainer.build_model()
    
    # Entrenar
    history = trainer.train(X_train, y_train, X_test, y_test, epochs=50)
    
    # Guardar modelo
    trainer.save_model()
    
    print("✅ Proceso completado exitosamente")
    
    return trainer.model


if __name__ == "__main__":
    train_model()