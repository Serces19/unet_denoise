import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

# Asegúrate de que model.py esté en la misma carpeta o en el path de Python
from model import CopycatUNet # <-- Importamos la nueva clase de modelo unificada

def generate_blending_mask(window_size, device):
    """Genera una máscara de mezcla piramidal para una ventana."""
    mask = np.ones((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            dist_to_edge = min(i, j, window_size - 1 - i, window_size - 1 - j)
            mask[i, j] = float(dist_to_edge) / (window_size / 2.0)
    mask = np.clip(mask, 0, 1)
    return torch.from_numpy(mask).to(device).unsqueeze(0)

def inference(args):
    """Función principal para ejecutar la inferencia en una imagen de alta resolución."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    # --- 1. Cargar el Modelo (de forma dinámica) ---
    print(f"Cargando modelo con encoder '{args.encoder}' y {args.n_out_channels} canales de salida...")
    
    # Se construye la arquitectura correcta ANTES de cargar los pesos
    model = CopycatUNet(
        n_out_channels=args.n_out_channels,
        encoder_name=args.encoder
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Modelo cargado exitosamente.")

    # --- 1.5 Ajuste Automático de Ventana para DINOv2 ---
    window_size = args.window_size
    if args.encoder == 'dinov2':
        patch_size = 14 # Tamaño de parche de DINOv2
        if window_size % patch_size != 0:
            new_size = (window_size // patch_size) * patch_size
            if new_size == 0: new_size = patch_size
            print(f"ADVERTENCIA: El tamaño de ventana {window_size} no es múltiplo de {patch_size} para DINOv2.")
            print(f"             Ajustando al múltiplo inferior más cercano: {new_size}.")
            window_size = new_size

    # --- 2. Cargar y Pre-procesar la Imagen de Entrada ---
    print(f"Cargando la imagen de entrada: {args.input}")
    bgr_image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {args.input}")

    # Convertir a RGB. Asumimos que la entrada para la tarea es RGB.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb_image.shape
    print(f"Dimensiones de la imagen: {W}x{H}")

    # Normalizar la imagen RGB de [0, 255] a [-1, 1]
    input_tensor = torch.from_numpy(rgb_image.astype(np.float32)).permute(2, 0, 1)
    input_tensor = (input_tensor / 127.5) - 1.0
    input_tensor = input_tensor.to(device)

    # --- 3. Inferencia con Ventana Deslizante ---
    print(f"Iniciando inferencia con ventana de {window_size}x{window_size} y solapamiento de {args.overlap}...")
    
    # Los tensores de salida y pesos se adaptan al número de canales de salida
    output_image = torch.zeros(args.n_out_channels, H, W, device=device)
    weight_map = torch.zeros(1, H, W, device=device) # El mapa de pesos siempre es de 1 canal
    
    blending_mask = generate_blending_mask(window_size, device)
    stride = window_size - args.overlap
    
    num_patches_h = int(np.ceil((H - window_size) / stride)) + 1 if H > window_size else 1
    num_patches_w = int(np.ceil((W - window_size) / stride)) + 1 if W > window_size else 1

    with torch.no_grad():
        progress_bar = tqdm(total=num_patches_h * num_patches_w, desc="Procesando parches")
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = min(i * stride, H - window_size) if H > window_size else 0
                x = min(j * stride, W - window_size) if W > window_size else 0
                
                patch = input_tensor[:, y:y+window_size, x:x+window_size].unsqueeze(0)
                prediction = model(patch)
                
                output_image[:, y:y+window_size, x:x+window_size] += prediction[0] * blending_mask
                weight_map[:, y:y+window_size, x:x+window_size] += blending_mask
                
                progress_bar.update(1)
        progress_bar.close()

    final_image_tensor = output_image / (weight_map + 1e-8)

    # --- 4. Post-procesar y Guardar la Imagen de Salida ---
    print("Post-procesando y guardando la imagen final...")
    
    # La de-normalización depende de la tarea, asumimos que la salida es [-1, 1]
    # Si tu máscara está en [0, 1], deberías quitar esta línea.
    final_image_tensor = (final_image_tensor * 0.5 + 0.5)
    
    final_image_tensor = torch.clamp(final_image_tensor, 0, 1) * 255.0
    output_np = final_image_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    
    # Se convierte la salida a BGR para guardarla con OpenCV, sea de 1 o 3 canales
    if args.n_out_channels == 1:
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_GRAY2BGR)
    else:
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, output_bgr)
    print(f"¡Proceso completado! Imagen guardada en: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de inferencia flexible para modelos UNet (DINOv2 o Clásico).")
    
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pth del modelo entrenado.')
    parser.add_argument('--input', type=str, required=True, help='Ruta a la imagen de entrada RGB.')
    parser.add_argument('--output', type=str, default='output/result.png', help='Ruta para guardar la imagen de salida.')
    
    # Argumentos clave para la flexibilidad
    parser.add_argument('--encoder', type=str, default='dinov2', choices=['dinov2', 'classic'], help='Tipo de encoder con el que se entrenó el modelo.')
    parser.add_argument('--n_out_channels', type=int, default=1, help='Número de canales de salida del modelo (1 para máscaras/grises, 3 para color).')

    # Argumentos para la inferencia
    parser.add_argument('--window_size', type=int, default=756, help='Tamaño de la ventana deslizante. Se ajustará si usa DINOv2 y no es múltiplo de 14.')
    parser.add_argument('--overlap', type=int, default=128, help='Número de píxeles de solapamiento entre ventanas.')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo a usar para la inferencia (ej. "cuda", "cpu").')
    
    args = parser.parse_args()
    inference(args)