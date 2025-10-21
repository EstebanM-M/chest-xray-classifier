# cargar_modelo.py
# Script para cargar el modelo entrenado de clasificación de rayos X

import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

def cargar_modelo(ruta_modelo='chest_xray_efficientnet_b0.pt'):
    '''
    Carga el modelo entrenado desde un archivo .pt
    
    Args:
        ruta_modelo: Ruta al archivo .pt del modelo
        
    Returns:
        model: Modelo cargado y listo para inferencia
        checkpoint: Información adicional (clases, métricas, etc.)
    '''
    # Cargar checkpoint
    checkpoint = torch.load(ruta_modelo, map_location='cpu')
    
    # Recrear arquitectura del modelo
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)  # 3 clases
    
    # Cargar pesos entrenados
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Modo evaluación
    
    print("✅ Modelo cargado correctamente")
    print(f"📊 Clases: {checkpoint['classes']}")
    print(f"🎯 Test Accuracy: {checkpoint['test_acc']:.4f}")
    
    return model, checkpoint


def predecir_imagen(model, ruta_imagen, checkpoint):
    '''
    Hace predicción sobre una imagen de rayos X
    
    Args:
        model: Modelo cargado
        ruta_imagen: Ruta a la imagen de rayos X
        checkpoint: Información del modelo (incluye clases)
        
    Returns:
        clase_predicha: Nombre de la clase predicha
        probabilidades: Probabilidades para cada clase
    '''
    # Transformaciones (igual que en entrenamiento)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar y preprocesar imagen
    imagen = Image.open(ruta_imagen).convert('RGB')
    imagen_tensor = transform(imagen).unsqueeze(0)  # Añadir dimensión batch
    
    # Predecir
    with torch.no_grad():
        output = model(imagen_tensor)
        probabilidades = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = output.argmax(1).item()
    
    clase_predicha = checkpoint['classes'][pred_idx]
    
    return clase_predicha, probabilidades.numpy()


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("CARGADOR DE MODELO - CLASIFICADOR DE RAYOS X")
    print("=" * 60)
    
    # Cargar modelo
    model, checkpoint = cargar_modelo('chest_xray_efficientnet_b0.pt')
    
    print("\n📋 Información del modelo:")
    print(f"  • Arquitectura: EfficientNet-B0")
    print(f"  • Número de clases: {len(checkpoint['classes'])}")
    print(f"  • Clases: {', '.join(checkpoint['classes'])}")
    
    # Ejemplo de predicción (descomenta si tienes una imagen)
    # clase, probs = predecir_imagen(model, 'mi_rayo_x.jpg', checkpoint)
    # print(f"\nPredicción: {clase}")
    # print("\nProbabilidades:")
    # for i, clase_nombre in enumerate(checkpoint['classes']):
    #     print(f"  {clase_nombre}: {probs[i]:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ Modelo listo para usar")
    print("=" * 60)
