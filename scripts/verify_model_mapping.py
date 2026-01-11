"""
Verifica el mapeo del modelo
"""

import numpy as np
import os
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model
from utils.dataset import LABEL_MAP, print_label_map, preprocess_char_unified

print("="*70)
print("🔍 VERIFICANDO MODELO Y MAPEO")
print("="*70)

# Mostrar mapeo
print_label_map()

# Cargar modelo
model = load_model('models/trained_model.h5')

print("\n📊 INFORMACIÓN DEL MODELO:")
print(f"   Input shape: {model.input_shape}")
print(f"   Output shape: {model.output_shape}")
print(f"   Número de clases: {model.output_shape[-1]}")

# Crear imagen de prueba con letra 'P'
print("\n🧪 PRUEBA CON IMAGEN DE REFERENCIA:")

# Buscar carpeta de mayúsculas
mayusculas_paths = [
    Path('data/referencia/Mayúsculas'),
    Path('data/referencia/Mayusculas'),
    Path('data/dataset/DATASET_IA')
]

test_img_path = None
for base_path in mayusculas_paths:
    if base_path.exists():
        # Buscar archivos que empiecen con P
        p_files = list(base_path.glob('P*.png')) + list(base_path.glob('P*.jpg'))
        if p_files:
            test_img_path = str(p_files[0])
            break
        
        # Si no hay archivos con P, buscar cualquier imagen
        all_files = list(base_path.glob('*.png')) + list(base_path.glob('*.jpg'))
        if all_files:
            test_img_path = str(all_files[0])
            expected_char = Path(all_files[0]).stem[0]  # Primera letra del nombre
            break

if test_img_path:
    print(f"   Usando: {test_img_path}")
    
    # Cargar imagen con manejo de errores
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Si falla, intentar con lectura en modo binario
    if img is None:
        print("   ⚠️  Error al leer con cv2.imread, intentando método alternativo...")
        with open(test_img_path, 'rb') as f:
            img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        print(f"   ✅ Imagen cargada: {img.shape}")
        
        # Procesar
        img_processed = preprocess_char_unified(img, debug=True)
        img_input = img_processed.reshape(1, 32, 32, 1)
        
        # Predecir
        prediction = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_char = LABEL_MAP.get(predicted_class, '?')
        
        # Detectar carácter esperado del nombre del archivo
        filename = Path(test_img_path).stem
        expected_char = filename[0] if filename else '?'
        
        print(f"\n📝 RESULTADO:")
        print(f"   Archivo: {Path(test_img_path).name}")
        print(f"   Esperado: '{expected_char}'")
        print(f"   Clase predicha: {predicted_class}")
        print(f"   Carácter predicho: '{predicted_char}'")
        print(f"   Confianza: {confidence:.1%}")
        print(f"   ¿Correcto? {'✅ SÍ' if predicted_char == expected_char else '❌ NO'}")
        
        # Mostrar top 5
        top5_indices = np.argsort(prediction[0])[-5:][::-1]
        print(f"\n   Top 5 predicciones:")
        for idx in top5_indices:
            char = LABEL_MAP.get(idx, '?')
            conf = prediction[0][idx]
            print(f"      {idx:3d} → '{char}' : {conf:.1%}")
    else:
        print(f"   ❌ No se pudo cargar la imagen: {test_img_path}")

else:
    print("   ❌ No se encontraron imágenes de prueba")
    print("\n   Carpetas buscadas:")
    for path in mayusculas_paths:
        print(f"      - {path} {'✅' if path.exists() else '❌'}")

print("\n" + "="*70)
print("🧪 PRUEBA ADICIONAL: Predicción sintética")
print("="*70)

# Crear una imagen sintética con una 'A'
print("\n   Creando imagen sintética de 'A'...")
synthetic = np.zeros((32, 32), dtype=np.uint8)

# Dibujar una 'A' simple
# Línea izquierda
cv2.line(synthetic, (8, 28), (16, 8), 255, 2)
# Línea derecha
cv2.line(synthetic, (16, 8), (24, 28), 255, 2)
# Línea horizontal
cv2.line(synthetic, (12, 20), (20, 20), 255, 2)

print(f"   Imagen creada: {synthetic.shape}")

# Procesar
synthetic_processed = preprocess_char_unified(synthetic, debug=True)
synthetic_input = synthetic_processed.reshape(1, 32, 32, 1)

# Predecir
prediction = model.predict(synthetic_input, verbose=0)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)
predicted_char = LABEL_MAP.get(predicted_class, '?')

print(f"\n📝 RESULTADO:")
print(f"   Esperado: 'A' (índice 10)")
print(f"   Clase predicha: {predicted_class}")
print(f"   Carácter predicho: '{predicted_char}'")
print(f"   Confianza: {confidence:.1%}")
print(f"   ¿Correcto? {'✅ SÍ' if predicted_char == 'A' else '❌ NO'}")

# Mostrar top 5
top5_indices = np.argsort(prediction[0])[-5:][::-1]
print(f"\n   Top 5 predicciones:")
for idx in top5_indices:
    char = LABEL_MAP.get(idx, '?')
    conf = prediction[0][idx]
    print(f"      {idx:3d} → '{char}' : {conf:.1%}")

print("\n" + "="*70)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*70)