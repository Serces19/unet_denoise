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
from datetime import datetime

# Importamos nuestras clases personalizadas
# Asumimos la estructura de carpetas: src/utils/logger.py
from model import CopycatUNet
from dataset import PairedImageDataset 
from logger import Logger

def train_fn(args):
    """Función principal de entrenamiento, ahora agnóstica a la tarea."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")
    
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    # --- INICIALIZAR LOGGER con un nombre de ejecución único ---
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_full_name = f"{run_timestamp}_{args.run_name}_{args.encoder}"
    logger = Logger(base_log_dir="../runs", run_name=run_full_name)
    print(f"Nombre de la ejecución: {run_full_name}")

    # --- 1. CARGADORES DE DATOS ---
    print(f"Configurando dataset para la tarea: '{args.run_name}'...")
    num_images = len(os.listdir(args.input_dir))
    indices = list(range(num_images))
    split_point = int(args.val_split * num_images)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_point:], indices[:split_point]

    # Usamos el PairedImageDataset genérico
    train_dataset = PairedImageDataset(
        input_dir=args.input_dir, 
        gt_dir=args.gt_dir, 
        crop_size=args.crop_size, 
        augment=True,
        advanced_augment=args.advanced_augment, # <-- AÑADIDO
        target_mode=args.target_mode
    )
    train_subset = Subset(train_dataset, train_indices)
    
    val_dataset = PairedImageDataset(
        input_dir=args.input_dir, 
        gt_dir=args.gt_dir, 
        crop_size=args.crop_size, 
        augment=False,
        advanced_augment=False, # <-- La aumentación avanzada NUNCA se aplica a la validación
        target_mode=args.target_mode
    )
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Dataset completo con {num_images} imágenes.")
    print(f" -> {len(train_subset)} para entrenamiento, {len(val_subset)} para validación.")

    # --- 2. INICIALIZAR MODELO, PÉRDIDA Y OPTIMIZADOR ---
    print(f"Inicializando modelo con encoder '{args.encoder}' y {args.n_out_channels} canales de salida...")
    model = CopycatUNet(
        n_out_channels=args.n_out_channels, 
        encoder_name=args.encoder,
        dino_model_name=args.dino_model_name
    ).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss() if args.loss_fn == 'bce' else nn.L1Loss()
    print(f"Usando pérdida: {args.loss_fn.upper()}")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=(device.type == 'cuda'))

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
        
        for input_img, target_img in train_loop:
            input_img, target_img = input_img.to(device), target_img.to(device)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                predicted_output = model(input_img)
                loss = loss_fn(predicted_output, target_img)

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
            for input_img, target_img in val_loader:
                input_img, target_img = input_img.to(device), target_img.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    predicted_output = model(input_img)
                    loss = loss_fn(predicted_output, target_img)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.5f}, Val Loss = {avg_val_loss:.5f}")

        logger.log_scalar('Loss/train', avg_train_loss, epoch + 1)
        logger.log_scalar('Loss/val', avg_val_loss, epoch + 1)

        # GUARDAR EL MEJOR MODELO
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(args.model_output_dir, f"best_model_{args.run_name}.pth")
            print(f"  -> Nueva mejor pérdida. Guardando checkpoint en {model_save_path}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'args': args
            }
            torch.save(checkpoint, model_save_path)

    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float))}
    metrics = {'best_validation_loss': best_val_loss}
    logger.log_hparams(hparams, metrics)
    logger.close()
    print("\nEntrenamiento finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework de entrenamiento flexible para tareas de Imagen-a-Imagen.")
    
    # --- Argumentos para definir la TAREA ---
    parser.add_argument('--run_name', type=str, required=True, help='Nombre para esta ejecución (ej. "deaging_classic", "matte_dinov2").')
    parser.add_argument('--input_dir', type=str, required=True, help='Directorio de imágenes de entrada.')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directorio de imágenes objetivo (ground truth).')
    parser.add_argument('--target_mode', type=str, default='RGB', choices=['RGB', 'L'], help="Modo del objetivo: 'RGB' para color, 'L' para escala de grises/máscaras.")
    parser.add_argument('--n_out_channels', type=int, default=3, help='Canales de salida del modelo (3 para RGB, 1 para máscaras).')
    
    # --- Argumentos de Arquitectura y Entrenamiento ---
    parser.add_argument('--encoder', type=str, default='classic', choices=['dinov2', 'classic'])
    parser.add_argument('--dino_model_name', type=str, default='dinov2_vits14', help='Modelo DINOv2 a usar.')
    parser.add_argument('--loss_fn', type=str, default='l1', choices=['bce', 'l1'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--resume_from', type=str, default=None, help='Ruta al checkpoint para reanudar.')
    parser.add_argument('--advanced_augment', action='store_true', help='Activa un set de aumentaciones de datos más agresivas.')
    parser.add_argument('--model_output_dir', type=str, default=str(Path(__file__).resolve().parent.parent / "models"))
    
    args = parser.parse_args()
    train_fn(args)