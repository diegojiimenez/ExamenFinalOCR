"""
Arquitectura del modelo CNN para reconocimiento de caracteres
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_character_cnn(config):
    """
    Construye una CNN simple para clasificación de caracteres.
    
    Args:
        config (dict): Configuración del modelo
        
    Returns:
        keras.Model: Modelo compilado
    """
    input_shape = config['model']['input_shape']
    num_classes = config['model']['num_classes']
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Capa de normalización
        layers.Rescaling(1./255),
        
        # Bloque Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque Convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config['model']['training']['learning_rate']
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_augmentation(config):
    """
    Crea pipeline de data augmentation.
    """
    aug_config = config['model']['training']['augmentation']
    
    if not aug_config['enabled']:
        return None
    
    return keras.Sequential([
        layers.RandomRotation(aug_config['rotation_range'] / 180.0),
        layers.RandomTranslation(
            height_factor=aug_config['height_shift'],
            width_factor=aug_config['width_shift']
        ),
        layers.RandomZoom(aug_config['zoom_range']),
    ])