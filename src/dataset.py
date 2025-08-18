import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ColorizationDataset(Dataset):
    """
    Dataset para la tarea de colorización.
    Carga imágenes de un directorio, las convierte a escala de grises para la entrada
    y mantiene la versión a color como el objetivo.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directorio con todas las imágenes.
            transform (callable, optional): Transformaciones opcionales a aplicar.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Cargar la imagen de forma robusta a caracteres especiales
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Asegurarse de que la imagen se cargó correctamente
        if bgr_image is None:
            print(f"ADVERTENCIA: No se pudo cargar la imagen: {img_path}")
            # Devolver tensores vacíos o manejar el error como se prefiera
            return torch.empty(3, 252, 252), torch.empty(3, 252, 252)

        # Convertir a RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # --- Crear Input y Target ---
        
        # Target: La imagen a color original
        target_image = rgb_image

        # Input: La imagen en escala de grises
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # Replicar el canal gris 3 veces para que DINOv2 la acepte (espera 3 canales)
        input_image = np.stack([gray_image]*3, axis=-1)

        # Aplicar transformaciones si existen
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# --- Transformaciones Estándar ---
dino_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((252, 252), antialias=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normaliza a [-1, 1]
])

# --- Bloque de prueba ---
if __name__ == '__main__':
    # 1. Crear un directorio y una imagen de prueba
    print("Creando un directorio de datos de prueba: 'data/colorization_test'")
    os.makedirs("../data/colorization_test", exist_ok=True)
    test_image_path = "../data/colorization_test/test_img.png"
    
    # Crear una imagen de prueba simple (un cuadrado de color)
    dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
    dummy_image[50:200, 50:200, 0] = 255 # Canal Rojo
    dummy_image[100:150, 100:150, 1] = 255 # Canal Verde
    dummy_image[125:175, 125:175, 2] = 255 # Canal Azul
    cv2.imwrite(test_image_path, cv2.cvtColor(dummy_image, cv2.COLOR_RGB2BGR))
    print(f"Imagen de prueba guardada en: {test_image_path}")

    # 2. Instanciar el Dataset
    print("\nInstanciando el Dataset...")
    color_dataset = ColorizationDataset(root_dir="../data/colorization_test", transform=dino_transform)
    print(f"Dataset encontrado con {len(color_dataset)} imagen(es).")

    # 3. Obtener una muestra y verificar las dimensiones
    if len(color_dataset) > 0:
        input_tensor, target_tensor = color_dataset[0]
        
        print("\n--- Verificación de la Muestra ---")
        print(f"Forma del tensor de entrada (gris, 3ch): {input_tensor.shape}")
        print(f"Forma del tensor objetivo (color):      {target_tensor.shape}")
        print(f"Tipo de dato de los tensores: {input_tensor.dtype}")
        print(f"Valor mínimo del tensor: {input_tensor.min():.2f}")
        print(f"Valor máximo del tensor: {input_tensor.max():.2f}")
        
        # Comprobar que la normalización se aplicó
        assert input_tensor.shape == (3, 252, 252), "La forma del tensor es incorrecta!"
        assert target_tensor.shape == (3, 252, 252), "La forma del tensor es incorrecta!"
        # Definimos una pequeña tolerancia para las comparaciones de punto flotante
        epsilon = 1e-6
        assert input_tensor.min() >= -1.0 - epsilon and input_tensor.max() <= 1.0 + epsilon, "La normalización falló"
        print("\n¡La verificación del dataset fue exitosa!")
