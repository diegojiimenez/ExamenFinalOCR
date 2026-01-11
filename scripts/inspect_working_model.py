"""
Inspecciona el modelo que SÍ funciona para adaptar el código
"""

from tensorflow.keras.models import load_model
import numpy as np

print("\n" + "="*60)
print("🔍 INSPECCIONANDO MODELO FUNCIONAL")
print("="*60)

model = load_model('models/trained_model.h5')

print("\n📊 ARQUITECTURA:")
print("="*60)
model.summary()

print("\n📋 INFORMACIÓN:")
print("="*60)
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Número de clases: {model.output_shape[-1]}")

# Intentar predecir con una imagen de prueba
print("\n🧪 PRUEBA DE PREDICCIÓN:")
print("="*60)

# Crear imagen de prueba (28x28 con ruido aleatorio)
test_img = np.random.rand(1, 28, 28, 1).astype('float32')
try:
    prediction = model.predict(test_img, verbose=0)
    print(f"✅ Predicción exitosa")
    print(f"   Shape de salida: {prediction.shape}")
    print(f"   Clases predichas: {prediction.shape[-1]}")
except Exception as e:
    print(f"❌ Error: {e}")

print("="*60)