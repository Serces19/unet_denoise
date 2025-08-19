# file: src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

# Importamos nuestras clases personalizadas
# Asumimos que este script está en la carpeta 'src/'
from model import CopycatUNet
from dataset import RedChannelDataset 
from logger import Logger # Importamos el logger para TensorBoard

def train_fn(args):
    """Función principal que ejecuta el bucle de entrenamiento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")
    
    # Crear directorios de salida si no existen
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    # --- INICIALIZAR EL LOGGER DE TENSORBOARD ---
    logger = Logger(base_log_dir="../runs")

    # --- 1. CARGADORES DE DATOS ---
    # La lógica de transformaciones ahora está dentro de RedChannelDataset
    print("Configurando los datasets...")
    num_images = len(os.listdir(args.rgb_dir))
    indices = list(range(num_images))
    split_point = int(args.val_split * num_images)
    
    np.random.seed(42) # Semilla para reproducibilidad
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_point:], indices[:split_point]

    # Dataset de entrenamiento CON aumentación (augment=True)
    train_dataset = RedChannelDataset(rgb_dir=args.rgb_dir, mask_dir=args.mask_dir, crop_size=args.crop_size, augment=True)
    train_subset = Subset(train_dataset, train_indices)
    
    # Dataset de validación SIN aumentación (augment=False)
    val_dataset = RedChannelDataset(rgb_dir=args.rgb_dir, mask_dir=args.mask_dir, crop_size=args.crop_size, augment=False)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset completo con {num_images} imágenes.")
    print(f" -> {len(train_subset)} para entrenamiento, {len(val_subset)} para validación.")

    # --- 2. INICIALIZAR MODELO, PÉRDIDA Y OPTIMIZADOR ---
    print(f"Inicializando modelo con encoder '{args.encoder}'...")
    model = CopycatUNet(n_out_channels=1, encoder_name=args.encoder).to(device)
    
    if args.loss_fn.lower() == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
        print("Usando pérdida: BCEWithLogitsLoss (recomendado para máscaras)")
    else:
        loss_fn = nn.L1Loss()
        print("Usando pérdida: L1Loss")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler() # Para entrenamiento con precisión mixta

    # --- 3. BUCLE DE ENTRENAMIENTO ---
    best_val_loss = float('inf')
    print(f"\nIniciando entrenamiento...")
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        # Nombres de variables corregidos: input_rgb, target_mask
        for input_rgb, target_mask in train_loop:
            input_rgb = input_rgb.to(device)
            target_mask = target_mask.to(device)
            
            # Forward pass con precisión mixta
            with torch.cuda.amp.autocast():
                predicted_output = model(input_rgb)
                loss = loss_fn(predicted_output, target_mask)
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Bucle de Validación ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_rgb, target_mask in val_loader:
                input_rgb = input_rgb.to(device)
                target_mask = target_mask.to(device)
                predicted_output = model(input_rgb)
                loss = loss_fn(predicted_output, target_mask)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Val Loss = {avg_val_loss:.5f}")

        # --- REGISTRAR MÉTRICAS CON EL LOGGER ---
        logger.log_scalar('Loss/train', avg_train_loss, epoch + 1)
        logger.log_scalar('Loss/val', avg_val_loss, epoch + 1)

        # --- 4. GUARDAR EL MEJOR MODELO (CHECKPOINTING) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.model_output_dir, f"best_model_{args.encoder}.pth")
            print(f"  -> Nueva mejor pérdida. Guardando modelo en {model_save_path}")
            torch.save(model.state_dict(), model_save_path)

    # --- REGISTRO FINAL DE HIPERPARÁMETROS ---
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float))}
    metrics = {'best_validation_loss': best_val_loss}
    logger.log_hparams(hparams, metrics)
    logger.close()
    print("\nEntrenamiento finalizado.")

if __name__ == "__main__":
    # Usamos argparse para configurar el entrenamiento desde la línea de comandos
    parser = argparse.ArgumentParser(description="Entrenar modelo UNet para Matte/Channel Generation.")
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    parser.add_argument('--rgb_dir', type=str, default=str(BASE_DIR.parent / "data" / "mate" / "rgb"), help='Directorio de imágenes RGB.')
    parser.add_argument('--mask_dir', type=str, default=str(BASE_DIR.parent / "data" / "mate" / "mask"), help='Directorio de máscaras.')
    parser.add_argument('--model_output_dir', type=str, default=str(BASE_DIR.parent / "models"), help='Directorio para guardar modelos.')
    
    # Hyperparameters
    parser.add_argument('--encoder', type=str, default='classic', choices=['dinov2', 'classic'], help='Tipo de encoder a usar.')
    parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce', 'l1'], help='Función de pérdida (bce recomendado para máscaras).')
    parser.add_argument('--num_epochs', type=int, default=100, help='Número de épocas.')
    parser.add_argument('--batch_size', type=int, default=8, help='Tamaño del lote.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Tasa de aprendizaje.')
    parser.add_argument('--crop_size', type=int, default=392, help='Tamaño del recorte (debe ser múltiplo de 14 para DINOv2).')
    parser.add_argument('--val_split', type=float, default=0.15, help='Porcentaje de datos para validación.')
    
    args = parser.parse_args()
    
    train_fn(args)