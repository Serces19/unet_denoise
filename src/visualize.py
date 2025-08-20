import torch
import torchvision

def denormalize(tensor):
    """
    De-normaliza un tensor de imagen del rango [-1, 1] al rango [0, 1].
    Asume una normalización de mean=0.5, std=0.5.
    """
    return tensor * 0.5 + 0.5

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
    # De-normalizar todas las imágenes para visualización correcta ([0, 1])
    input_tensors = denormalize(input_tensors)
    
    # El target puede ser de 1 canal (L) o 3 (RGB)
    if target_tensors.shape[1] == 1:
        # Repetir el canal gris 3 veces para que se pueda apilar con los otros
        target_tensors = target_tensors.repeat(1, 3, 1, 1)
    else:
        target_tensors = denormalize(target_tensors)

    # La predicción podría tener una activación sigmoid (0 a 1) o ser logits.
    # Si usa BCEWithLogitsLoss, los logits están en cualquier rango, así que aplicamos sigmoid.
    # Si usa L1, la salida ya está en [-1, 1].
    if predicted_tensors.min() < -0.1: # Asumimos que es una salida de L1
        predicted_tensors = denormalize(predicted_tensors)
    else: # Asumimos que son logits de BCE
        predicted_tensors = torch.sigmoid(predicted_tensors)

    if predicted_tensors.shape[1] == 1:
        predicted_tensors = predicted_tensors.repeat(1, 3, 1, 1)

    # Asegurarse de que todos los tensores estén en la CPU para hacer el mosaico
    combined_grid = torch.cat([
        input_tensors.cpu(), 
        target_tensors.cpu(), 
        predicted_tensors.cpu()
    ], dim=0)

    # Crear el mosaico
    # nrow es el número de imágenes por fila. Aquí mostramos el lote completo,
    # con Input arriba, GT en medio y Predicción abajo.
    grid = torchvision.utils.make_grid(combined_grid, nrow=input_tensors.size(0))
    
    # Añadir la imagen al logger de TensorBoard
    tag = f'{tag_prefix}_Images_Epoch_{epoch}'
    logger.writer.add_image(tag, grid, global_step=epoch)
    print(f" -> Imágenes de muestra guardadas en TensorBoard con la etiqueta: {tag}")