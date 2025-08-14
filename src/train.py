import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm

# Importamos nuestras clases personalizadas
from model import SimpleCopyCat
from dataset import ColorizationDataset

# --- CONFIGURACIÓN Y HIPERPARÁMETROS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4 # Ajustar según la VRAM disponible. 4 es un valor seguro para empezar.
NUM_EPOCHS = 25 # Número de épocas para el entrenamiento

# Directorios (usamos los datos de prueba generados por dataset.py por ahora)
DATA_DIR = "data/colorization_test"
MODEL_OUTPUT_DIR = "models/"

def train_fn():
    """Función principal que ejecuta el bucle de entrenamiento."""
    print(f"Usando el dispositivo: {DEVICE}")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 1. CARGADORES DE DATOS (DATA LOADERS)
    # Definimos las mismas transformaciones que usamos para probar el dataset
    dino_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = ColorizationDataset(root_dir=DATA_DIR, transform=dino_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset cargado con {len(train_dataset)} imágenes.")

    # 2. INICIALIZAR MODELO, PÉRDIDA Y OPTIMIZADOR
    model = SimpleCopyCat(n_out_channels=3, fine_tune_encoder=True).to(DEVICE)
    # Para regresión de imágenes, L1Loss (Error Absoluto Medio) es un buen punto de partida
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. BUCLE DE ENTRENAMIENTO
    print("\nIniciando el entrenamiento...")
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        total_loss = 0.0

        for batch_idx, (input_gray, target_color) in enumerate(loop):
            input_gray = input_gray.to(DEVICE)
            target_color = target_color.to(DEVICE)

            # Forward pass
            generated_color = model(input_gray)
            loss = loss_fn(generated_color, target_color)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f"Fin de la Época {epoch+1} | Pérdida Promedio: {avg_loss:.4f}")

    # 4. GUARDAR EL MODELO
    print("\nEntrenamiento finalizado. Guardando el modelo...")
    model_save_path = os.path.join(MODEL_OUTPUT_DIR, "baseline_colorizer.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo guardado en: {model_save_path}")

if __name__ == "__main__":
    # Advertencia para el usuario si no existen los datos de prueba
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("ADVERTENCIA: El directorio de datos 'data/colorization_test' está vacío o no existe.")
        print("Por favor, ejecuta 'python src/dataset.py' primero para generar datos de prueba.")
    else:
        train_fn()
