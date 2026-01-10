"""
Entrenamiento del modelo OCR con arquitectura CNN mejorada
ARQUITECTURA AJUSTADA AL MODELO PRE-ENTRENADO (32x32, 62 clases)
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
    
    def __init__(self, img_size=32):  # 🔄 CAMBIADO: 28 → 32
        self.img_size = img_size
        self.char_map = self._build_char_map()
        self.reverse_char_map = {v: k for k, v in self.char_map.items()}
        self.num_classes = len(self.char_map)
        self.model = None
    
    def _build_char_map(self):
        """Construye el mapa de caracteres desde configuración"""
        # 62 clases: A-Z (26) + a-z (26) + 0-9 (10)
        char_map_str = config.get('model.char_map', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        return {i: char for i, char in enumerate(char_map_str)}
    
    def load_dataset(self):
        """
        Carga el dataset desde las carpetas configuradas
        """
        print("\n" + "="*60)
        print("📦 CARGANDO DATASET")
        print("="*60)
        
        all_images = data_loader.get_dataset_images('all')
        
        if not all_images:
            raise ValueError("⚠️  No se encontraron imágenes en los datasets")
        
        print(f"📊 Total de imágenes encontradas: {len(all_images)}")
        print("⏳ Procesando imágenes (esto puede tardar)...\n")
        
        X = []
        y = []
        skipped = 0
        
        total = len(all_images)
        for idx, img_path in enumerate(all_images):
            if (idx + 1) % 1000 == 0:
                print(f"   Procesadas: {idx + 1}/{total} ({(idx+1)/total*100:.1f}%)")
            
            try:
                label = self._extract_label(img_path)
                if label is None:
                    skipped += 1
                    continue
                
                img_path_str = str(img_path)
                img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    with open(img_path_str, 'rb') as f:
                        img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
                
                if img is None or img.size == 0:
                    skipped += 1
                    continue
                
                # 🔧 IMPORTANTE: Redimensionar a 32x32
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_normalized = img_resized.astype('float32') / 255.0
                
                X.append(img_normalized)
                y.append(label)
                
            except Exception:
                skipped += 1
        
        print(f"\n✅ Procesamiento completado: {len(X)}/{total} imágenes válidas")
        
        if not X:
            raise ValueError("❌ No se pudieron cargar imágenes válidas del dataset")
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape para CNN: (N, 32, 32, 1)
        X = X.reshape(-1, self.img_size, self.img_size, 1)
        
        print(f"\n📊 Resumen:")
        print(f"   ✅ Cargadas: {len(X)} imágenes")
        print(f"   ⚠️  Omitidas: {skipped}")
        
        # Split train/test
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"\n🔹 Train: {len(X_train)} muestras")
        print(f"🔹 Test: {len(X_test)} muestras")
        print("="*60 + "\n")
        
        return X_train, X_test, y_train, y_test
    
    def _extract_label(self, img_path):
        """Extrae la etiqueta de la ruta"""
        img_path = Path(img_path)
        
        try:
            filename = img_path.stem
            parent_name = img_path.parent.name
            
            # Método 1: Carpeta padre
            if len(parent_name) == 1 and parent_name in self.reverse_char_map:
                return self.reverse_char_map[parent_name]
            
            # Método 2: Primer carácter del nombre
            if filename and filename[0] in self.reverse_char_map:
                return self.reverse_char_map[filename[0]]
            
            # Método 3: Buscar en el nombre
            for char in filename:
                if char in self.reverse_char_map:
                    return self.reverse_char_map[char]
        except:
            pass
        
        return None
    
    def build_model(self):
        """
        Construye la arquitectura CNN EXACTA del modelo pre-entrenado
        """
        print("🏗️  Construyendo arquitectura del modelo...")
        
        model = Sequential([
            # Primera capa convolucional (padding='same' para mantener 32x32)
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(self.img_size, self.img_size, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Segunda capa convolucional (padding='same')
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Tercera capa convolucional (padding='same')
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten: (4, 4, 128) → 2048
            Flatten(),
            
            # Capas densas
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu'),
            Dropout(0.5),
            
            # Capa de salida: 62 clases
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("✅ Modelo construido exitosamente")
        print(f"📊 Total de parámetros: {model.count_params():,}")
        print(f"📋 Número de clases: {self.num_classes}")
        
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Entrena el modelo"""
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("="*60)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('models/trained_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        ]
        
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
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Precisión en test: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"📉 Pérdida en test: {test_loss:.4f}")
        print("="*60 + "\n")
        
        return history
    
    def save_model(self, filepath='models/trained_model.h5'):
        """Guarda el modelo entrenado"""
        os.makedirs('models', exist_ok=True)
        self.model.save(filepath)
        print(f"💾 Modelo guardado en: {filepath}")


def train_model():
    """Función principal para entrenar el modelo"""
    print("\n" + "🔤"*30)
    print("SISTEMA DE ENTRENAMIENTO OCR")
    print("🔤"*30 + "\n")
    
    try:
        trainer = OCRModelTrainer()
        X_train, X_test, y_train, y_test = trainer.load_dataset()
        trainer.build_model()
        history = trainer.train(X_train, y_train, X_test, y_test, epochs=50)
        trainer.save_model()
        
        print("\n✅ Proceso completado exitosamente")
        print("🎉 El modelo está listo para usar")
        
        return trainer.model
    
    except Exception as e:
        print(f"\n❌ ERROR durante el entrenamiento:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    train_model()