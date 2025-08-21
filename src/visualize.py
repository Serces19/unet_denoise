import torch
import torchvision

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    De-normaliza un tensor de imagen.
    Asume que el tensor de entrada está en el rango [-1, 1].
    Lo convierte al rango [0, 1] para visualización.
    """
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean

def save_batch_for_tensorboard(input_tensors, target_tensors, predicted_tensors, logger, epoch, tag_prefix="Validation"):
    """
    Crea un mosaico de imágenes (input, target, prediction) y lo guarda en TensorBoard.
    
    Args:
        input_tensors (torch.Tensor): Lote de imágenes de entrada.
        target_tensors (torch.Tensor): Lote de imágenes objetivo (ground truth).
        predicted_tensors (torch.Tensor): Lote de imágenes predichas por el modelo.
        logger (Logger): Instancia de tu clase Logger.
        epoch (int): Época actual.
        tag_prefix (str): Prefijo para la etiqueta en TensorBoard (ej. 'Validation' o 'Training').
    """
    # Asegurarse de que todos los tensores estén en la CPU para hacer el mosaico
    input_tensors = input_tensors.cpu()
    target_tensors = target_tensors.cpu()
    predicted_tensors = predicted_tensors.cpu()
    
    # De-normalizar las imágenes de entrada para una visualización correcta
    input_tensors = denormalize(input_tensors)
    
    # Manejar el objetivo (target)
    if target_tensors.shape[1] == 1: # Si es una máscara/escala de grises
        target_tensors = target_tensors.repeat(1, 3, 1, 1) # Repetir para que sea RGB
    else: # Si es una imagen RGB
        target_tensors = denormalize(target_tensors)
        
    # Manejar la predicción
    # Si la pérdida es BCE, la salida son logits. Aplicamos Sigmoid para verlos como imagen.
    # Si la pérdida es L1, la salida ya está en [-1, 1] y necesita ser de-normalizada.
    if predicted_tensors.min() < 0:
        predicted_tensors = denormalize(predicted_tensors)
    else:
        predicted_tensors = torch.sigmoid(predicted_tensors)

    if predicted_tensors.shape[1] == 1:
        predicted_tensors = predicted_tensors.repeat(1, 3, 1, 1)

    # Combinar los lotes en una sola lista para el mosaico:
    # Fila 1: Entradas, Fila 2: Objetivos (Ground Truth), Fila 3: Predicciones
    combined_tensors = torch.cat([input_tensors, target_tensors, predicted_tensors], dim=0)

    # Crear el mosaico de imágenes
    grid = torchvision.utils.make_grid(combined_tensors, nrow=input_tensors.size(0))
    
    # Añadir la imagen al logger de TensorBoard
    tag = f'A_Images/{tag_prefix}'
    logger.writer.add_image(tag, grid, global_step=epoch)
    print(f" -> Imágenes de muestra de '{tag_prefix}' guardadas en TensorBoard.")