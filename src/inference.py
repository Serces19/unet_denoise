# file: src/inference.py

import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
from pathlib import Path

# Asegúrate de que model.py esté en la misma carpeta o en el path de Python
from model import CopycatUNet

def generate_blending_mask(window_size, device):
    """Genera una máscara de mezcla piramidal para una ventana."""
    mask = np.ones((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            dist_to_edge = min(i, j, window_size - 1 - i, window_size - 1 - j)
            mask[i, j] = float(dist_to_edge) / (window_size / 2.0)
    mask = np.clip(mask, 0, 1)
    return torch.from_numpy(mask).to(device).unsqueeze(0)

def process_single_image(model, image_path, window_size, overlap, n_out_channels, device):
    """
    Toma un modelo cargado y la ruta a una imagen, y devuelve la imagen procesada.
    """
    try:
        bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            print(f"ADVERTENCIA: No se pudo cargar la imagen, saltando: {image_path}")
            return None
    except Exception as e:
        print(f"ADVERTENCIA: Error al leer el archivo {image_path}: {e}")
        return None

    # --- Pre-procesamiento ---
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb_image.shape
    input_tensor = torch.from_numpy(rgb_image.astype(np.float32)).permute(2, 0, 1)
    input_tensor = (input_tensor / 127.5) - 1.0
    input_tensor = input_tensor.to(device)

    # --- Inferencia con Ventana Deslizante ---
    output_image = torch.zeros(n_out_channels, H, W, device=device)
    weight_map = torch.zeros(1, H, W, device=device)
    blending_mask = generate_blending_mask(window_size, device)
    stride = window_size - overlap # Corregido para usar el parámetro 'overlap'
    
    num_patches_h = int(np.ceil((H - window_size) / stride)) + 1 if H > window_size else 1
    num_patches_w = int(np.ceil((W - window_size) / stride)) + 1 if W > window_size else 1

    with torch.no_grad():
        patch_bar = tqdm(total=num_patches_h * num_patches_w, desc=f"Procesando {Path(image_path).name}", leave=False)
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = min(i * stride, H - window_size) if H > window_size else 0
                x = min(j * stride, W - window_size) if W > window_size else 0
                patch = input_tensor[:, y:y+window_size, x:x+window_size].unsqueeze(0)
                prediction = model(patch)
                output_image[:, y:y+window_size, x:x+window_size] += prediction[0] * blending_mask
                weight_map[:, y:y+window_size, x:x+window_size] += blending_mask
                patch_bar.update(1)
        patch_bar.close()

    final_image_tensor = output_image / (weight_map + 1e-8)

    # --- Post-procesamiento ---
    final_image_tensor = (final_image_tensor * 0.5 + 0.5)
    final_image_tensor = torch.clamp(final_image_tensor, 0, 1) * 255.0
    output_np = final_image_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    
    if n_out_channels == 1:
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_GRAY2BGR)
    else:
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
    return output_bgr

def main(args):
    """
    Función principal que carga el modelo y distribuye el trabajo de inferencia.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    # --- 1. Cargar el Modelo (una sola vez) ---
    print(f"Cargando modelo '{args.model_size}' con encoder '{args.encoder}' (DINO: {args.dino_model_name if args.encoder == 'dinov2' else 'N/A'})...")
    
    # <-- MODIFICADO: Pasamos el argumento model_size al constructor
    model = CopycatUNet(
        n_out_channels=args.n_out_channels,
        encoder_name=args.encoder,
        dino_model_name=args.dino_model_name,
        model_size=args.model_size
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Modelo cargado exitosamente.")

    # --- Ajuste de Ventana para DINOv2 ---
    window_size = args.window_size
    if args.encoder == 'dinov2':
        patch_size = 14
        if window_size % patch_size != 0:
            new_size = (window_size // patch_size) * patch_size
            if new_size == 0: new_size = patch_size
            print(f"ADVERTENCIA: El tamaño de ventana {window_size} no es múltiplo de {patch_size} para DINOv2.")
            print(f"             Ajustando al múltiplo inferior más cercano: {new_size}.")
            window_size = new_size

    # --- 2. Determinar si la entrada es un archivo o una carpeta ---
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        print(f"Procesando archivo individual: {input_path}")
        result_bgr = process_single_image(model, input_path, window_size, args.overlap, args.n_out_channels, device)
        if result_bgr is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result_bgr)
            print(f"¡Proceso completado! Imagen guardada en: {output_path}")

    elif input_path.is_dir():
        print(f"Procesando todos los archivos en la carpeta: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        files_to_process = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
        
        file_bar = tqdm(files_to_process, desc="Procesando archivos")
        for file_path in file_bar:
            result_bgr = process_single_image(model, file_path, window_size, args.overlap, args.n_out_channels, device)
            if result_bgr is not None:
                save_path = output_path / file_path.name
                cv2.imwrite(str(save_path), result_bgr)
        print(f"\n¡Proceso completado! Todas las imágenes han sido guardadas en: {output_path}")

    else:
        raise FileNotFoundError(f"La ruta de entrada no existe o no es un archivo/carpeta válido: {args.input}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de inferencia flexible para procesar un archivo o una carpeta.")
    
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pth del modelo entrenado.')
    parser.add_argument('--input', type=str, required=True, help='Ruta a la imagen de entrada O a una carpeta con imágenes.')
    parser.add_argument('--output', type=str, required=True, help='Ruta al archivo de salida O a la carpeta de salida.')
    
    parser.add_argument('--encoder', type=str, default='classic', choices=['dinov2', 'classic'], help='Tipo de encoder con el que se entrenó el modelo.')
    parser.add_argument('--dino_model_name', type=str, default='dinov2_vits14', help='Nombre específico del modelo DINOv2 a usar (ej. dinov2_vitb14).')
    
    # <-- AÑADIDO: Argumento para seleccionar el tamaño del modelo clásico
    parser.add_argument('--model_size', type=str, default='medium', choices=['small', 'medium', 'big'], help="Tamaño de la UNet clásica (ignorado si el encoder es dinov2).")
    
    parser.add_argument('--n_out_channels', type=int, default=3, help='Número de canales de salida del modelo (1 para máscaras, 3 para color).')
    parser.add_argument('--window_size', type=int, default=768, help='Tamaño de la ventana deslizante. Se ajustará si usa DINOv2 y no es múltiplo de 14.')
    parser.add_argument('--overlap', type=int, default=128, help='Número de píxeles de solapamiento entre ventanas.')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo a usar para la inferencia (ej. "cuda", "cpu").')
    
    args = parser.parse_args()
    main(args)