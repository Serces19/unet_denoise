import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from tqdm import tqdm
from torch.utils.data import Subset

# Importamos nuestras clases personalizadas
from model import CopycatUNet
from dataset import ColorizationDataset, RedChannelDataset

# --- CONFIGURACIÓN Y HIPERPARÁMETROS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 8 # Se puede aumentar un poco al usar crops más pequeños
NUM_EPOCHS = 50 # Aumentamos las épocas ya que el data augmentation hace el aprendizaje más robusto

# Nuevos parámetros
CROP_SIZE = 224 # Tamaño estándar para muchos modelos de visión
VAL_SPLIT = 0.15 # 15% de los datos para validación

# Directorios
INPUT_DIR = "../data/mate/rgb"
GT_DIR = "../data/mate/mask"
MODEL_OUTPUT_DIR = "../models/"

def train_fn():
    """Función principal que ejecuta el bucle de entrenamiento."""
    print(f"Usando el dispositivo: {DEVICE}")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 1. TRANSFORMACIONES CON "AUTOCROPPING"
    # Transformación para entrenamiento (con data augmentation)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CROP_SIZE), # Redimensiona el lado más corto a CROP_SIZE
        transforms.RandomCrop(CROP_SIZE), # Recorta una región aleatoria de 224x224
        transforms.RandomHorizontalFlip(p=0.5), # Volteo horizontal aleatorio
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Transformación para validación (determinista, sin aumentación aleatoria)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(CROP_SIZE), # Redimensiona el lado más corto a CROP_SIZE
        transforms.CenterCrop(CROP_SIZE), # Recorta el centro de la imagen
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. CARGADORES DE DATOS (Lógica correcta para segmentación)
    # Ya no definimos train_transform y val_transform aquí.
    # Esa lógica ahora está encapsulada dentro de RedChannelDataset.

    # 1. Obtenemos y barajamos los índices de los archivos PRIMERO.
    #    Así decidimos qué imágenes van a entrenamiento y cuáles a validación
    #    antes de crear cualquier objeto Dataset.
    num_images = len(os.listdir(INPUT_DIR)) # Asumiendo que RGB_DIR está definido
    indices = list(range(num_images))
    split_point = int(VAL_SPLIT * num_images)

    np.random.seed(42) # Usamos una semilla para que la división sea reproducible
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_point:], indices[:split_point]

    # 2. Creamos un Dataset de ENTRENAMIENTO configurado CON aumentación de datos.
    train_dataset_instance = RedChannelDataset(
        rgb_dir=INPUT_DIR, 
        mask_dir=GT_DIR, 
        crop_size=CROP_SIZE, 
        augment=True  # <-- La aumentación está activada
    )
    # Creamos el subconjunto (Subset) usando solo los índices de entrenamiento.
    train_subset = Subset(train_dataset_instance, train_indices)

    # 3. Creamos un Dataset de VALIDACIÓN configurado SIN aumentación de datos.
    val_dataset_instance = RedChannelDataset(
        rgb_dir=INPUT_DIR, 
        mask_dir=GT_DIR, 
        crop_size=CROP_SIZE, 
        augment=False # <-- La aumentación está desactivada
    )
    # Creamos el subconjunto (Subset) usando solo los índices de validación.
    val_subset = Subset(val_dataset_instance, val_indices)

    # 4. Creamos los DataLoaders a partir de los Subsets.
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset completo con {num_images} imágenes.")
    print(f" -> {len(train_subset)} imágenes para entrenamiento.")
    print(f" -> {len(val_subset)} imágenes para validación.")

    # 3. INICIALIZAR MODELO, PÉRDIDA Y OPTIMIZADOR
    model = CopycatUNet(n_out_channels=1, fine_tune_encoder=True).to(DEVICE)
    model = CopycatUNet(
                        n_out_channels=1,         # O 3, dependiendo de tu tarea
                        encoder_name='dinov2',    # <-- Elige el encoder aquí
                        fine_tune_encoder=True
                        ).to(DEVICE)
    loss_fn = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)



    # 4. BUCLE DE ENTRENAMIENTO
    best_val_loss = float('inf')
    print("\nIniciando el entrenamiento...")
    for epoch in range(NUM_EPOCHS):
        model.train() # Poner el modelo en modo entrenamiento
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train")
        total_train_loss = 0.0

        for batch_idx, (input_gray, target_color) in enumerate(train_loop):
            input_gray, target_color = input_gray.to(DEVICE), target_color.to(DEVICE)
            
            generated_color = model(input_gray)
            loss = loss_fn(generated_color, target_color)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Bucle de Validación ---
        model.eval() # Poner el modelo en modo evaluación
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val", leave=False)
        total_val_loss = 0.0
        with torch.no_grad():
            for input_gray, target_color in val_loop:
                input_gray, target_color = input_gray.to(DEVICE), target_color.to(DEVICE)
                generated_color = model(input_gray)
                loss = loss_fn(generated_color, target_color)
                total_val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # 5. GUARDAR EL MEJOR MODELO (CHECKPOINTING)
        if avg_val_loss < best_val_loss:
            print(f"  -> Nueva mejor pérdida de validación. Guardando modelo en {MODEL_OUTPUT_DIR}")
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(MODEL_OUTPUT_DIR, "best_colorizer.pth")
            torch.save(model.state_dict(), model_save_path)

    print("\nEntrenamiento finalizado.")
    print(f"El mejor modelo fue guardado en: {os.path.join(MODEL_OUTPUT_DIR, 'best_colorizer.pth')}")

if __name__ == "__main__":
    import numpy as np
    train_fn()