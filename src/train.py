# file: src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Importamos nuestras clases personalizadas
from model import CopycatUNet
from dataset import PairedImageDataset 
from losses import HybridLoss
from logger import Logger
from visualize import save_batch_for_tensorboard

def train_fn(args):
    """
    Función principal de entrenamiento con scheduler, modelos escalables y visualización mejorada.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")
    
    os.makedirs(args.model_output_dir, exist_ok=True)
    
    # --- INICIALIZAR LOGGER ---
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_full_name = f"{run_timestamp}_{args.run_name}_{args.encoder}"
    logger = Logger(base_log_dir="../runs", run_name=run_full_name)
    print(f"Nombre de la ejecución: {run_full_name}")

    # --- 1. CARGADORES DE DATOS ---
    print(f"Configurando dataset para la tarea: '{args.run_name}'...")
    image_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    num_images = len(image_files)
    all_indices = list(range(num_images))
    
    if args.val_split > 0:
        print(f"Dividiendo dataset: {(1-args.val_split)*100:.0f}% train, {args.val_split*100:.0f}% val.")
        split_point = int(args.val_split * num_images); np.random.seed(42); np.random.shuffle(all_indices)
        train_indices, val_indices = all_indices[split_point:], all_indices[:split_point]
        train_dataset = PairedImageDataset(input_dir=args.input_dir, gt_dir=args.gt_dir, crop_size=args.crop_size, augment=True, advanced_augment=args.advanced_augment, target_mode=args.target_mode)
        train_subset = Subset(train_dataset, train_indices)
        val_dataset = PairedImageDataset(input_dir=args.input_dir, gt_dir=args.gt_dir, crop_size=args.crop_size, augment=False, advanced_augment=False, target_mode=args.target_mode)
        val_subset = Subset(val_dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True); val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        print(f"Dataset: {num_images} imgs -> {len(train_subset)} train, {len(val_subset)} val.")
    else:
        print("ADVERTENCIA: No se usará validación. Entrenando con todos los datos."); train_dataset = PairedImageDataset(input_dir=args.input_dir, gt_dir=args.gt_dir, crop_size=args.crop_size, augment=True, advanced_augment=args.advanced_augment, target_mode=args.target_mode); train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True); val_loader = None
        print(f"Dataset: {num_images} imgs -> {len(train_dataset)} train, 0 val.")

    # --- 2. INICIALIZAR MODELO, PÉRDIDA, OPTIMIZADOR Y SCHEDULER ---
    print(f"Inicializando modelo '{args.model_size}' con encoder '{args.encoder}'...")
    model = CopycatUNet(n_out_channels=args.n_out_channels, encoder_name=args.encoder, 
                        dino_model_name=args.dino_model_name, model_size=args.model_size).to(device)
    
    if args.loss_fn.lower() == 'hybrid':
        loss_fn = HybridLoss(device=device, n_channels=args.n_out_channels, w_l1=args.w_l1, w_perceptual=args.w_perceptual, w_laplacian=args.w_laplacian, w_ssim=args.w_ssim)
    elif args.loss_fn.lower() == 'bce': loss_fn = nn.BCEWithLogitsLoss()
    else: loss_fn = nn.L1Loss()
    print(f"Usando pérdida: {args.loss_fn.upper()}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)

    # --- LÓGICA PARA CARGAR UN CHECKPOINT ---
    start_epoch = 0; best_val_loss = float('inf')
    if args.resume_from and os.path.isfile(args.resume_from):
        print(f"=> Reanudando entrenamiento desde: {args.resume_from}"); checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict']); optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1; best_val_loss = checkpoint.get('loss', float('inf'))
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"=> Checkpoint cargado. Reanudando desde la época {start_epoch}")
    elif args.resume_from: print(f"ADVERTENCIA: No se encontró checkpoint en '{args.resume_from}'.")

    # --- 3. BUCLE DE ENTRENAMIENTO ---
    print(f"\nIniciando entrenamiento desde la época {start_epoch}...")
    for epoch in range(start_epoch, args.num_epochs):
        model.train(); train_loss_components = {}; first_train_batch = None
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}] Train")
        
        for i, (input_img, target_img) in enumerate(train_loop):
            if i == 0: first_train_batch = (input_img, target_img)
            input_img, target_img = input_img.to(device), target_img.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                predicted_output = model(input_img); loss_output = loss_fn(predicted_output, target_img)
            
            if isinstance(loss_output, dict):
                loss = loss_output['total']; 
                for k, v in loss_output.items(): train_loss_components.setdefault(k, 0.0); train_loss_components[k] += v.item()
                train_loop.set_postfix({k: f"{v.item():.4f}" for k, v in loss_output.items() if 'total' not in k})
            else:
                loss = loss_output; train_loss_components.setdefault('total', 0.0); train_loss_components['total'] += loss.item()
                train_loop.set_postfix(loss=loss.item())
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()

        print(f"\nEpoch {epoch+1} Train Loss:", end=" "); 
        for key, value in train_loss_components.items():
            avg_loss = value / len(train_loader); logger.log_scalar(f'Loss-Train/{key}', avg_loss, epoch + 1); print(f"{key}={avg_loss:.5f}", end=" | ")
        
        if val_loader is not None:
            model.eval(); val_loss_components = {}; last_val_batch = None
            with torch.no_grad():
                for input_img, target_img in val_loader:
                    last_val_batch = (input_img, target_img)
                    input_img, target_img = input_img.to(device), target_img.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        predicted_output = model(input_img); loss_output = loss_fn(predicted_output, target_img)
                    if isinstance(loss_output, dict):
                        for k, v in loss_output.items(): val_loss_components.setdefault(k, 0.0); val_loss_components[k] += v.item()
                    else:
                        val_loss_components.setdefault('total', 0.0); val_loss_components['total'] += loss_output.item()

            print(f"Val Loss:", end=" "); 
            for key, value in val_loss_components.items():
                avg_loss = value / len(val_loader); logger.log_scalar(f'Loss-Val/{key}', avg_loss, epoch + 1); print(f"{key}={avg_loss:.5f}", end=" | ")
            print()
            
            avg_val_loss = val_loss_components['total'] / len(val_loader)
            scheduler.step(avg_val_loss)
            logger.log_scalar('meta/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss; model_save_path = os.path.join(args.model_output_dir, f"best_model_{args.run_name}.pth")
                print(f"  -> Nueva mejor pérdida de validación. Guardando checkpoint en {model_save_path}")
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': best_val_loss, 'args': vars(args)}
                torch.save(checkpoint, model_save_path)
        else:
             print()
        
        if epoch == 0 or (epoch + 1) % args.viz_epochs == 0:
            print("Generando visualizaciones para TensorBoard..."); model.eval()
            if first_train_batch is not None:
                with torch.no_grad():
                    input_img_viz, target_img_viz = first_train_batch; predicted_output_for_viz = model(input_img_viz.to(device))
                save_batch_for_tensorboard(input_img_viz, target_img_viz, predicted_output_for_viz, logger, epoch + 1, tag_prefix="Training_Augmented")
            if val_loader is not None and last_val_batch is not None:
                with torch.no_grad():
                    input_img_viz, target_img_viz = last_val_batch; predicted_output_for_viz = model(input_img_viz.to(device))
                save_batch_for_tensorboard(input_img_viz, target_img_viz, predicted_output_for_viz, logger, epoch + 1, tag_prefix="Validation_Clean")

    if val_loader is None:
        model_save_path = os.path.join(args.model_output_dir, f"final_model_epoch_{args.num_epochs}_{args.run_name}.pth")
        print(f"\nEntrenamiento sin validación finalizado. Guardando modelo final en {model_save_path}")
        checkpoint = {'epoch': args.num_epochs - 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': -1, 'args': vars(args)}
        torch.save(checkpoint, model_save_path)
    
    hparams = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float))}; metrics = {'best_validation_loss': best_val_loss if val_loader is not None else -1}
    logger.log_hparams(hparams, metrics); logger.close(); print("\nEntrenamiento finalizado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Framework de entrenamiento flexible para tareas de Imagen-a-Imagen.")
    
    parser.add_argument('--run_name', type=str, required=True); parser.add_argument('--input_dir', type=str, required=True); parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--target_mode', type=str, default='RGB', choices=['RGB', 'L']); parser.add_argument('--n_out_channels', type=int, default=3)
    
    parser.add_argument('--encoder', type=str, default='classic', choices=['dinov2', 'classic'])
    parser.add_argument('--dino_model_name', type=str, default='dinov2_vits14')
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'big'], help="Tamaño de la UNet clásica.")
    
    parser.add_argument('--loss_fn', type=str, default='hybrid', choices=['bce', 'l1', 'hybrid']); 
    parser.add_argument('--w_l1', type=float, default=1.0); parser.add_argument('--w_perceptual', type=float, default=0.1)
    parser.add_argument('--w_laplacian', type=float, default=0.5); parser.add_argument('--w_ssim', type=float, default=0.25)

    parser.add_argument('--num_epochs', type=int, default=100); parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4); parser.add_argument('--crop_size', type=int, default=392)
    parser.add_argument('--val_split', type=float, default=0.15, help='Poner a 0 para desactivar validación.')
    parser.add_argument('--resume_from', type=str, default=None); parser.add_argument('--advanced_augment', action='store_true')
    parser.add_argument('--viz_epochs', type=int, default=10, help='Frecuencia para guardar imágenes en TensorBoard.')
    parser.add_argument('--model_output_dir', type=str, default=str(Path(__file__).resolve().parent.parent / "models"))
    
    args = parser.parse_args()
    train_fn(args)