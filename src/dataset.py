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
        """
        Args:
            advanced_augment (bool): Si es True, aplica aumentaciones más agresivas.
        """
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.crop_size = crop_size
        self.augment = augment
        self.advanced_augment = advanced_augment # <-- Nuevo parámetro
        self.target_mode = target_mode
        
        self.input_files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
        
        print(f"Dataset creado. Modo de objetivo: {target_mode}. Aumentación básica: {augment}. Aumentación avanzada: {advanced_augment}.")
        print(f"Se encontraron {len(self.input_files)} imágenes de entrada en: {input_dir}")

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
            # ... (código de manejo de errores sin cambios)
            return torch.empty(3, self.crop_size, self.crop_size), torch.empty(3 if self.target_mode == 'RGB' else 1, self.crop_size, self.crop_size)

        # --- APLICAR TRANSFORMACIONES ---
        
        # 1. Aumentaciones geométricas (aplicadas a ambos, input y gt)
        if self.augment:
            # Volteo horizontal básico
            if random.random() > 0.5:
                input_image = TF.hflip(input_image)
                gt_image = TF.hflip(gt_image)

            # Aumentaciones geométricas avanzadas (opcionales)
            if self.advanced_augment:
                # Rotación aleatoria
                angle = transforms.RandomRotation.get_params([-10, 10])
                input_image = TF.rotate(input_image, angle, interpolation=TF.InterpolationMode.BICUBIC)
                gt_image = TF.rotate(gt_image, angle, interpolation=TF.InterpolationMode.BICUBIC)
        
        # 2. Redimensionar y Recortar (Zoom)
        # RandomResizedCrop es una forma potente de hacer zoom y recortar a la vez.
        # Si no hay aumentación, se comporta como un Resize + CenterCrop.
        if self.augment and self.advanced_augment:
            # Zoom aleatorio entre 80% y 100%
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                input_image, scale=(0.8, 1.0), ratio=(0.95, 1.05)
            )
            input_image = TF.resized_crop(input_image, i, j, h, w, [self.crop_size, self.crop_size], interpolation=TF.InterpolationMode.BICUBIC)
            gt_image = TF.resized_crop(gt_image, i, j, h, w, [self.crop_size, self.crop_size], interpolation=TF.InterpolationMode.BICUBIC)
        else:
            # Comportamiento estándar: redimensionar y recortar el centro
            input_image = TF.resize(input_image, self.crop_size, interpolation=TF.InterpolationMode.BICUBIC)
            gt_image = TF.resize(gt_image, self.crop_size, interpolation=TF.InterpolationMode.BICUBIC)
            input_image = TF.center_crop(input_image, [self.crop_size, self.crop_size])
            gt_image = TF.center_crop(gt_image, [self.crop_size, self.crop_size])


        # 3. Aumentaciones fotométricas (solo aplicadas al INPUT)
        if self.augment and self.advanced_augment:
            # "Grade" - Ajustes de color
            input_image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(input_image)
            # "Defocus" - Desenfoque Gaussiano
            if random.random() > 0.5:
                 input_image = input_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
            
                # "Occlusion" - Borrado aleatorio
            random_erasing = transforms.RandomErasing(
                p=0.22, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=False
            )
            input_image = transforms.ToTensor()(input_image)  # RandomErasing espera tensor
            input_image = random_erasing(input_image)
            input_image = transforms.ToPILImage()(input_image)  # Convertir de nuevo si lo necesitas


        # 4. Conversión a Tensor y Normalización
        input_tensor = TF.to_tensor(input_image)
        target_tensor = TF.to_tensor(gt_image)
        
        # "Ruido" - Añadido al tensor de entrada
        if self.augment and self.advanced_augment:
            noise = torch.randn_like(input_tensor) * 0.05 # Ruido gaussiano pequeño
            input_tensor = torch.clamp(input_tensor + noise, 0, 1)

        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        input_tensor = normalizer(input_tensor)
        
        if self.target_mode == 'RGB':
            target_tensor = normalizer(target_tensor)

        return input_tensor, target_tensor