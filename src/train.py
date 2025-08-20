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
# Asumimos que logger.py está en src/utils/
from model import CopycatUNet
from dataset import RedChannelDataset 
from logger import Logger

def train_fn(args):
    """Función principal que ejecuta el bucle de entrenamiento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")
    
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    logger = Logger(base_log_dir="../runs")

    # --- 1. CARGADORES DE DATOS ---
    print("Configurando los datasets...")
    num_images = len(os.listdir(args.rgb_dir))
    indices = list(range(num_images))
    split_point = int(args.val_split * num_images)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_point:], indices[:split_point]
    train_dataset = RedChannelDataset(rgb_dir=args.rgb_dir, mask_dir=args.mask_dir, crop_size=args.crop_size, augment=True)
    train_subset = Subset(train_dataset, train_indices)
    val_dataset = RedChannelDataset(rgb_dir=args.rgb_dir, mask_dir=args.mask_dir, crop_size=args.crop_size, augment=False)
    val_subset = Subset(val_dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset completo con {num_images} imágenes.")
    print(f" -> {len(train_subset)} para entrenamiento, {len(val_subset)} para validación.")

    # --- 2. INICIALIZAR MODELO, PÉRDIDA Y OPTIMIZADOR ---
    print(f"Inicializando modelo con encoder '{args.encoder}' (DINO Model: {args.dino_model_name if args.encoder == 'dinov2' else 'N/A'})...")
    
    # MODIFICADO: Pasamos el nombre del modelo DINOv2 a nuestro CopycatUNet
    model = CopycatUNet(
        n_out_channels=1, 
        encoder_name=args.encoder,
        dino_model_name=args.dino_model_name
    ).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss() if args.loss_fn == 'bce' else nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # MODIFICADO: Corregido para nuevas versiones de PyTorch
    scaler = torch.amp.GradScaler(device_type='cuda', enabled=(device.type == 'cuda'))

    # --- LÓGICA PARA CARGAR UN CHECKPOINT ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.isfile(args.resume_from):
        print(f"=> Reanudando entrenamiento desde el checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"=> Checkpoint cargado. Se reanudará desde la época {start_epoch}")
    elif args.resume_from:
        print(f"ADVERTENCIA: No se encontró el checkpoint en '{args.resume_from}'. Empezando desde cero.")

    # --- 3. BUCLE DE ENTRENAMIENTO ---
    print(f"\nIniciando entrenamiento desde la época {start_epoch}...")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for input_rgb, target_mask in train_loop:
            input_rgb, target_mask = input_rgb.to(device), target_mask.to(device)
            
            # MODIFICADO: Corregido para nuevas versiones de PyTorch
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                predicted_output = model(input_rgb)
                loss = loss_fn(predicted_output, target_mask)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Bucle de Validación
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_rgb, target_mask in val_loader:
                input_rgb, target_mask = input_rgb.to(device), target_mask.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    predicted_output = model(input_rgb)
                    loss = loss_fn(predicted_output, target_mask)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Val Loss = {avg_val_loss:.5f}")

        logger.log_scalar('Loss/train', avg_train_loss, epoch + 1)
        logger.log_scalar('Loss/val', avg_val_loss, epoch + 1)

        # GUARDAR EL MEJOR MODELO
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.model_output_dir, f"best_model_{args.encoder}_{args.dino_model_name if args.encoder == 'dinov2' else ''}.pth")
            print(f"  -> Nueva mejor pérdida. Guardando checkpoint en {model_save_path}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'encoder': args.encoder,
                'dino_model_name': args.dino_model_name
            }
            torch.save(checkpoint, model_save_path)

    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float))}
    metrics = {'best_validation_loss': best_val_loss}
    logger.log_hparams(hparams, metrics)
    logger.close()
    print("\nEntrenamiento finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo UNet para Matte/Channel Generation.")
    
    BASE_DIR = Path(__file__).resolve().parent
    parser.add_argument('--rgb_dir', type=str, default=str(BASE_DIR.parent / "data" / "mate" / "rgb"))
    parser.add_argument('--mask_dir', type=str, default=str(BASE_DIR.parent / "data" / "mate" / "mask"))
    parser.add_argument('--model_output_dir', type=str, default=str(BASE_DIR.parent / "models"))
    
    parser.add_argument('--encoder', type=str, default='classic', choices=['dinov2', 'classic'])
    # MODIFICADO: Nuevo argumento para seleccionar el modelo DINOv2
    parser.add_argument('--dino_model_name', type=str, default='dinov2_vits14', help='Nombre específico del modelo DINOv2 a usar (ej. dinov2_vitb14).')
    
    parser.add_argument('--loss_fn', type=str, default='bce', choices=['bce', 'l1'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--resume_from', type=str, default=None)
    
    args = parser.parse_args()
    train_fn(args)