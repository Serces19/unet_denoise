# file: src/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from pytorch_msssim import SSIM # <-- Nueva importación

# --- COMPONENTES DE PÉRDIDA INDIVIDUALES ---

class PerceptualLoss(nn.Module):
    """Calcula la pérdida perceptual (VGG Loss)."""
    def __init__(self, device='cuda'):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.feature_layers = [3, 8, 17, 26, 35]
        self.vgg_slices = nn.ModuleList([vgg[:i+1] for i in self.feature_layers])
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.loss_fn = nn.L1Loss()

    def forward(self, prediction, target):
        prediction = (prediction + 1) / 2
        target = (target + 1) / 2
        prediction = self.transform(prediction)
        target = self.transform(target)
        perceptual_loss = 0.0
        for vgg_slice in self.vgg_slices:
            pred_features = vgg_slice(prediction)
            target_features = vgg_slice(target)
            perceptual_loss += self.loss_fn(pred_features, target_features)
        return perceptual_loss

class LaplacianPyramidLoss(nn.Module):
    """Calcula la pérdida de la pirámide laplaciana para preservar bordes nítidos."""
    def __init__(self, max_levels=3, channels=3, device='cuda'):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.loss_fn = nn.L1Loss()
        kernel = self._build_gaussian_kernel(channels=channels, device=device)
        self.gaussian_conv = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, bias=False, groups=channels)
        self.gaussian_conv.weight.data = kernel
        self.gaussian_conv.weight.requires_grad = False

    def _build_gaussian_kernel(self, channels, device):
        ax = torch.arange(-2, 3, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="xy")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * 1.0**2))
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel.repeat(channels, 1, 1, 1).to(device)

    def laplacian_pyramid(self, img):
        pyramid = []
        current_img = img
        for _ in range(self.max_levels):
            blurred = self.gaussian_conv(current_img)
            laplacian = current_img - blurred
            pyramid.append(laplacian)
            current_img = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        pyramid.append(current_img)
        return pyramid

    def forward(self, prediction, target):
        prediction = (prediction + 1) / 2
        target = (target + 1) / 2
        pred_pyramid = self.laplacian_pyramid(prediction)
        target_pyramid = self.laplacian_pyramid(target)
        loss = 0
        for pred_level, target_level in zip(pred_pyramid, target_pyramid):
            loss += self.loss_fn(pred_level, target_level)
        return loss / self.max_levels

class SSIMLoss(nn.Module):
    """
    Calcula la pérdida de Similitud Estructural (SSIM).
    SSIM es una métrica de 0 a 1 (1 es perfecto). La pérdida es 1 - SSIM.
    """
    def __init__(self, data_range=1.0, channels=3):
        super(SSIMLoss, self).__init__()
        # Usamos win_size=7 para mejor rendimiento en imágenes más pequeñas/recortadas
        self.ssim_module = SSIM(data_range=data_range, size_average=True, channel=channels, win_size=7)

    def forward(self, prediction, target):
        # SSIM espera que las imágenes estén en el rango [0, data_range], en nuestro caso [0, 1]
        prediction = (prediction + 1) / 2
        target = (target + 1) / 2
        return 1.0 - self.ssim_module(prediction, target)


# --- CLASE DE PÉRDIDA HÍBRIDA PRINCIPAL ---

class HybridLoss(nn.Module):
    """
    Combina L1, Perceptual, Laplaciana y SSIM con pesos ajustables.
    """
    def __init__(self, device='cuda', n_channels=3, w_l1=1.0, w_perceptual=0.1, w_laplacian=0.5, w_ssim=0.25):
        super(HybridLoss, self).__init__()
        self.w_l1 = w_l1
        self.w_perceptual = w_perceptual
        self.w_laplacian = w_laplacian
        self.w_ssim = w_ssim
        
        self.l1_loss = nn.L1Loss()
        
        if self.w_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(device=device)
        if self.w_laplacian > 0:
            self.laplacian_loss = LaplacianPyramidLoss(channels=n_channels, device=device)
        if self.w_ssim > 0:
            self.ssim_loss = SSIMLoss(channels=n_channels)
        
        print(f"Pérdida Híbrida inicializada con pesos: L1={w_l1}, Perceptual={w_perceptual}, Laplacian={w_laplacian}, SSIM={w_ssim}")

    def forward(self, prediction, target):
        loss_l1 = self.l1_loss(prediction, target)
        
        total_loss = self.w_l1 * loss_l1
        loss_components = {'l1': loss_l1}

        if self.w_perceptual > 0:
            loss_perceptual = self.perceptual_loss(prediction, target)
            total_loss += self.w_perceptual * loss_perceptual
            loss_components['perceptual'] = loss_perceptual

        if self.w_laplacian > 0:
            loss_laplacian = self.laplacian_loss(prediction, target)
            total_loss += self.w_laplacian * loss_laplacian
            loss_components['laplacian'] = loss_laplacian

        if self.w_ssim > 0:
            loss_ssim = self.ssim_loss(prediction, target)
            total_loss += self.w_ssim * loss_ssim
            loss_components['ssim'] = loss_ssim
            
        loss_components['total'] = total_loss
        return loss_components