# Resultados del Entrenamiento

Visualizaciones generadas durante el entrenamiento del modelo de clasificación de rayos X de tórax.

## 📊 Archivos

### training_curves.png
Curvas de **loss** y **accuracy** durante el entrenamiento, mostrando:
- Evolución de train y validation a través de las épocas
- Resumen de métricas finales
- Información del entrenamiento (tiempo, épocas, early stopping)

**Resultados:**
- Train Accuracy: 78.36%
- Validation Accuracy: 76.64% 
- Test Accuracy: 77.66%

### confusion_matrix.png
Matriz de confusión del modelo en el **test set**. 

Muestra:
- Predicciones correctas por clase
- Errores de clasificación entre clases
- Distribución de predicciones

## 🎯 Interpretación

El modelo muestra:
- ✅ **Buena generalización**: Val accuracy similar a train accuracy (no overfitting)
- ✅ **Desempeño balanceado**: Accuracy consistente en las 3 clases
- ✅ **Resultados reproducibles**: Test accuracy confirma el desempeño en validación

## 📈 Métricas Clave

| Métrica | Train  | Validation | Test   |
|---------|--------|------------|--------|
| Accuracy| 78.36% |   76.64%   | 77.66% |
| Loss    | 32.83% |   35.68%   | 34.58%  |

*Ver `model_info.txt` en `/models` para métricas detalladas por clase.*

## 🔍 Observaciones

- El modelo no presenta overfitting significativo
- Data augmentation ayudó a la generalización
- Class weights compensaron el desbalance del dataset
- EfficientNet-B0 demostró ser efectivo para este problema

