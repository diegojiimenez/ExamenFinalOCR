"""
Gestión del modelo de deep learning
"""

import os
from tensorflow.keras.models import load_model


class ModelManager:
    """
    Maneja la carga y verificación del modelo
    """
    
    def __init__(self, model_path="models/trained_model.h5"): 
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """Carga el modelo desde disco"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"❌ Modelo no encontrado en: {self.model_path}\n"
                f"   Asegúrate de tener 'trained_model.h5' en la carpeta 'models/'\n"
                f"   Ejecuta: python main.py train"
            )
        
        print(f"📦 Cargando modelo desde: {self.model_path}")
        self.model = load_model(self.model_path)
        print("✅ Modelo cargado exitosamente")
        
        return self.model
    
    def get_model_info(self):
        """Obtiene información del modelo"""
        if self.model is None:
            return None
        
        return {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'trainable_params': self.model.count_params()
        }