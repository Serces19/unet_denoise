import torch
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

# Asegúrate de que model.py esté en la misma carpeta o en el path de Python
from model import SimpleCopyCat

def generate_blending_mask(window_size, device):
    """
    Genera una máscara de mezcla piramidal para una ventana.
    Los valores son más altos en el centro (1.0) y disminuyen hacia los bordes.
    Esto da más peso a las predicciones del centro de cada parche.
    """
    mask = np.ones((window_size, window_size), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            # Distancia al borde más cercano
            dist_to_edge = min(i, j, window_size - 1 - i, window_size - 1 - j)
            # Normalizar por la mitad del tamaño de la ventana
            mask[i, j] = float(dist_to_edge) / (window_size / 2.0)
            
    # Asegurarse de que el valor máximo sea 1.0 y recortar valores negativos
    mask = np.clip(mask, 0, 1)
    return torch.from_numpy(mask).to(device).unsqueeze(0)


def inference(args):
    """
    Función principal para ejecutar la inferencia en una imagen de alta resolución.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    # --- 1. Cargar el Modelo ---
    print("Cargando el modelo...")
    # Recrea la misma arquitectura que usaste para entrenar
    model = SimpleCopyCat(n_out_channels=3, fine_tune_encoder=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # Poner el modelo en modo de evaluación (muy importante)
    print("Modelo cargado exitosamente.")

    # --- 2. Cargar y Pre-procesar la Imagen de Entrada ---
    print(f"Cargando la imagen de entrada: {args.input}")
    # Cargar la imagen usando OpenCV (formato BGR)
    bgr_image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {args.input}")

    # Convertir a RGB y obtener la versión en escala de grises
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    H, W = gray_image.shape
    print(f"Dimensiones de la imagen: {W}x{H}")

    # Replicar el canal gris 3 veces y normalizar a [-1, 1]
    input_gray_3ch = np.stack([gray_image] * 3, axis=-1)
    input_tensor = torch.from_numpy(input_gray_3ch.astype(np.float32)).permute(2, 0, 1)
    # Normalización: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
    input_tensor = (input_tensor / 127.5) - 1.0
    input_tensor = input_tensor.to(device)

    # --- 3. Inferencia con Ventana Deslizante ---
    print(f"Iniciando inferencia con ventana de {args.window_size}x{args.window_size} y solapamiento de {args.overlap}...")
    
    # Tensores para almacenar el resultado final y los pesos de la mezcla
    output_image = torch.zeros_like(input_tensor)
    weight_map = torch.zeros_like(input_tensor)
    
    # Generar la máscara de mezcla
    blending_mask = generate_blending_mask(args.window_size, device)

    stride = args.window_size - args.overlap
    
    # Calcular el número de parches
    num_patches_h = int(np.ceil((H - args.window_size) / stride)) + 1
    num_patches_w = int(np.ceil((W - args.window_size) / stride)) + 1

    with torch.no_grad(): # Desactivar el cálculo de gradientes
        progress_bar = tqdm(total=num_patches_h * num_patches_w, desc="Procesando parches")
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = min(i * stride, H - args.window_size)
                x = min(j * stride, W - args.window_size)
                
                # Extraer el parche de la imagen de entrada
                patch = input_tensor[:, y:y+args.window_size, x:x+args.window_size].unsqueeze(0)
                
                # Ejecutar el modelo en el parche
                prediction = model(patch)
                
                # Añadir la predicción mezclada al lienzo de salida
                output_image[:, y:y+args.window_size, x:x+args.window_size] += prediction[0] * blending_mask
                weight_map[:, y:y+args.window_size, x:x+args.window_size] += blending_mask
                
                progress_bar.update(1)
        progress_bar.close()

    # Normalizar el resultado dividiendo por la suma de los pesos
    # Añadir un pequeño epsilon para evitar la división por cero
    final_image_tensor = output_image / (weight_map + 1e-8)

    # --- 4. Post-procesar y Guardar la Imagen de Salida ---
    print("Post-procesando y guardando la imagen final...")
    
    # De-normalizar de [-1, 1] a [0, 255]
    final_image_tensor = (final_image_tensor * 0.5 + 0.5) * 255.0
    # Recortar valores para asegurarse de que están en el rango [0, 255]
    final_image_tensor = torch.clamp(final_image_tensor, 0, 255)
    
    # Convertir de Tensor a NumPy array, y de (C, H, W) a (H, W, C)
    output_np = final_image_tensor.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    
    # Convertir de RGB a BGR para guardar con OpenCV
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

    # Guardar la imagen
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, output_bgr)
    print(f"¡Proceso completado! Imagen guardada en: {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de inferencia para el modelo de colorización.")
    
    parser.add_argument('--model_path', type=str, default='../models/best_colorizer.pth', help='Ruta al archivo .pth del modelo entrenado.')
    parser.add_argument('--input', type=str, required=True, help='Ruta a la imagen de entrada (puede ser a color o en escala de grises).')
    parser.add_argument('--output', type=str, default='output/result.png', help='Ruta para guardar la imagen colorizada de salida.')
    
    parser.add_argument('--window_size', type=int, default=768, help='Tamaño de la ventana deslizante. ¡AJUSTAR SEGÚN TU VRAM!')
    parser.add_argument('--overlap', type=int, default=128, help='Número de píxeles de solapamiento entre ventanas.')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo a usar para la inferencia (ej. "cuda", "cpu").')

    args = parser.parse_args()
    
    inference(args)