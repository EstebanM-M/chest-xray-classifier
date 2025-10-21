# Modelos Entrenados

## Modelo Principal

**Archivo:** `chest_xray_efficientnet_b0.pt` (no incluido en GitHub por tamaño)

### Descargar Modelo

El modelo entrenado está disponible en Google Drive.

**Tamaño:** ~45 MB

### Información del Modelo

Ver `model_info.txt` para detalles completos.

**Especificaciones:**
- **Arquitectura:** EfficientNet-B0 (pre-entrenado en ImageNet)
- **Clases:** normal, pneumonia, tuberculosis
- **Número de clases:** 3
- **Test Accuracy:** 77%
- **Train Accuracy:** 76.93%
- **Val Accuracy:** 77%

### Cómo Usar el Modelo

1. Descarga el archivo `chest_xray_efficientnet_b0.pt` desde Google Drive
2. Colócalo en esta carpeta `models/`
3. Ejecuta el script de carga:
```
python cargar_modelo.py
```

### Requisitos

- Python 3.8+
- PyTorch 2.0+
- torchvision
- Pillow

### Ejemplo de Uso
```
# Cargar modelo
from cargar_modelo import cargar_modelo, predecir_imagen

model, checkpoint = cargar_modelo('chest_xray_efficientnet_b0.pt')

# Hacer predicción
clase, probs = predecir_imagen(model, 'mi_rayo_x.jpg', checkpoint)
print(f"Predicción: {clase}")
```

### Nota Importante

Este modelo es solo para fines educativos y de investigación. No debe usarse como herramienta de diagnóstico médico sin supervisión profesional.
```
