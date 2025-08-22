# file: src/dataset.py

import os
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import numpy as np

class PairedImageDataset(Dataset):
    """
    Dataset genérico y flexible para tareas de traducción de imagen a imagen.
    VERSIÓN MEJORADA con aumentaciones avanzadas opcionales.
    """
    def __init__(self, input_dir, gt_dir, crop_size, augment=False, 
                advanced_augment=False, target_mode='RGB'):
        super(PairedImageDataset, self).__init__() # Buena práctica llamar al __init__ del padre
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.crop_size = crop_size
        self.augment = augment
        self.advanced_augment = advanced_augment
        self.target_mode = target_mode
        
        self.input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
        
        print(f"Dataset creado. Modo de objetivo: {target_mode}. Aumentación básica: {augment}. Aumentación avanzada: {advanced_augment}.")
        print(f"Se encontraron {len(self.input_files)} imágenes de entrada en: {input_dir}")

        # --- NUEVO: Definir transformaciones UNA SOLA VEZ aquí ---
        # Esto es mucho más eficiente que crearlas en cada __getitem__
        if self.augment and self.advanced_augment:
            # "Grade"
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            # "Defocus"
            self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
            # "Occlusion"
            self.random_erasing = transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)


    def __len__(self):
        return len(self.input_files)


    def __getitem__(self, idx):
        filename = self.input_files[idx]
        input_path = os.path.join(self.input_dir, filename)
        gt_path = os.path.join(self.gt_dir, filename)

        try:
            input_image = Image.open(input_path).convert("RGB")
            gt_image = Image.open(gt_path).convert(self.target_mode)
        except FileNotFoundError:
            return torch.empty(3, self.crop_size, self.crop_size), torch.empty(3 if self.target_mode == 'RGB' else 1, self.crop_size, self.crop_size)

        # --- 1. Aumentaciones Geométricas (sobre imágenes PIL) ---
        if self.augment:
            # Volteo Horizontal
            if random.random() > 0.5:
                input_image = TF.hflip(input_image)
                gt_image = TF.hflip(gt_image)
            # Rotación
            if self.advanced_augment:
                angle = transforms.RandomRotation.get_params([-15, 15])
                input_image = TF.rotate(input_image, angle, interpolation=TF.InterpolationMode.BICUBIC)
                gt_image = TF.rotate(gt_image, angle, interpolation=TF.InterpolationMode.BICUBIC)

        # --- 2. Redimensionar y Recortar (manejando el zoom) ---
        if self.augment and self.advanced_augment:
            # RandomResizedCrop maneja el zoom y el recorte en un paso
            i, j, h, w = transforms.RandomResizedCrop.get_params(input_image, scale=(0.8, 1.0), ratio=(0.95, 1.05))
            input_image = TF.resized_crop(input_image, i, j, h, w, [self.crop_size, self.crop_size], interpolation=TF.InterpolationMode.BICUBIC)
            gt_image = TF.resized_crop(gt_image, i, j, h, w, [self.crop_size, self.crop_size], interpolation=TF.InterpolationMode.BICUBIC)
        else:
            # Comportamiento estándar sin aumentación: redimensionar y recortar el centro
            interp_mode_gt = TF.InterpolationMode.NEAREST if self.target_mode == 'L' else TF.InterpolationMode.BICUBIC
            input_image = TF.resize(input_image, [self.crop_size, self.crop_size], interpolation=TF.InterpolationMode.BICUBIC)
            gt_image = TF.resize(gt_image, [self.crop_size, self.crop_size], interpolation=interp_mode_gt)

        # --- 3. Aumentaciones Fotométricas (sobre imágenes PIL, aplicadas a ambos) ---
        if self.augment and self.advanced_augment:
            # Para aplicar la misma transformación aleatoria a ambas imágenes, guardamos y restauramos el estado del generador de números aleatorios
            state = torch.get_rng_state()
            input_image = self.color_jitter(input_image)
            torch.set_rng_state(state)
            gt_image = self.color_jitter(gt_image)

            # Hacemos lo mismo para el desenfoque
            if random.random() > 0.5:
                state = torch.get_rng_state()
                input_image = self.gaussian_blur(input_image)
                torch.set_rng_state(state)
                gt_image = self.gaussian_blur(gt_image)

        # --- 4. Conversión a Tensor ---
        input_tensor = TF.to_tensor(input_image)
        target_tensor = TF.to_tensor(gt_image)
        
        # --- 5. Aumentaciones sobre Tensores (aplicadas a ambos) ---
        if self.augment and self.advanced_augment:
            # Ruido
            if random.random() < 0.2: # Probabilidad de 20%
                noise = torch.randn_like(input_tensor) * random.uniform(0.01, 0.05)
                input_tensor = torch.clamp(input_tensor + noise, 0, 1)
                target_tensor = torch.clamp(target_tensor + noise, 0, 1)
            
            # Oclusión (Random Erasing)
            if random.random() < 0.2: # Probabilidad de 20%
                # Obtenemos los parámetros una vez
                i, j, h, w, v = self.random_erasing.get_params(input_tensor, scale=(0.02, 0.2), ratio=(0.3, 3.3))
                # Aplicamos el mismo borrado a ambos
                input_tensor = TF.erase(input_tensor, i, j, h, w, v)
                target_tensor = TF.erase(target_tensor, i, j, h, w, v)

        # --- 6. Normalización Final ---
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = normalizer(input_tensor)
        
        if self.target_mode == 'RGB':
            target_tensor = normalizer(target_tensor)

        return input_tensor, target_tensor
        

        
