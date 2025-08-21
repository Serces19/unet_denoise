# file: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Bloques de Construcción Reutilizables ---
# (DoubleConv y DecoderBlock no necesitan cambios)

class DoubleConv(nn.Module):
    """Bloque de convolución doble (Conv -> BN -> ReLU) * 2, un pilar de la UNet."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DecoderBlock(nn.Module):
    """Bloque decoder de la U-Net."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

# --- Encoders Modulares ---

class UNetEncoder(nn.Module):
    """
    Encoder convolucional clásico de una U-Net.
    MODIFICADO: Ahora es escalable en anchura (small, medium, big).
    """
    def __init__(self, n_channels=3, model_size='medium'): # <-- NUEVO ARGUMENTO
        super().__init__()
        
        # --- LÓGICA DE ESCALADO ---
        # Definimos los canales base para cada tamaño de modelo
        if model_size == 'small':
            base_channels = 32
        elif model_size == 'medium':
            base_channels = 64
        elif model_size == 'big':
            base_channels = 96 # Un buen paso intermedio antes de duplicar
        else:
            raise ValueError(f"Tamaño de modelo '{model_size}' no reconocido. Elige 'small', 'medium', o 'big'.")

        # Generamos la lista de canales dinámicamente
        self.channels = [base_channels * (2**i) for i in range(5)] # ej. [64, 128, 256, 512, 1024]
        print(f"UNetEncoder Clásico creado (tamaño: {model_size}) con canales: {self.channels}")
        
        # La arquitectura se construye usando la lista de canales dinámica
        self.inc = DoubleConv(n_channels, self.channels[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[0], self.channels[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[1], self.channels[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[2], self.channels[3]))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[3], self.channels[4]))
        
    def forward(self, x):
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        return [s1, s2, s3, s4, s5]

class DinoV2Encoder(nn.Module):
    """Wrapper para el encoder DINOv2."""
    # (Esta clase no necesita cambios)
    def __init__(self, model_name='dinov2_vits14', fine_tune=True, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
        for param in self.dinov2.parameters():
            param.requires_grad = fine_tune

    def forward(self, x):
        features = self.dinov2.get_intermediate_layers(x, n=self.n_layers, return_class_token=False)
        patch_size = self.dinov2.patch_embed.patch_size[0]
        h = x.shape[2] // patch_size
        w = x.shape[3] // patch_size
        reshaped_features = []
        for f in features:
            dim = f.shape[-1]
            reshaped_f = f.permute(0, 2, 1).reshape(-1, dim, h, w)
            reshaped_features.append(reshaped_f)
        return reshaped_features

# --- Modelo Principal Flexible ---

class CopycatUNet(nn.Module):
    """
    Modelo principal que une un Encoder (a elección) con un Decoder tipo U-Net.
    MODIFICADO: Acepta el tamaño del modelo para el encoder clásico.
    """
    def __init__(self, n_out_channels=3, encoder_name='dinov2', fine_tune_encoder=True,
                 dino_model_name='dinov2_vits14', model_size='medium'): # <-- NUEVO ARGUMENTO
        super().__init__()
        
        if encoder_name.lower() == 'dinov2':
            self.encoder = DinoV2Encoder(model_name=dino_model_name, fine_tune=fine_tune_encoder, n_layers=4)
            if "vits14" in dino_model_name: dim = 384
            elif "vitb14" in dino_model_name: dim = 768
            elif "vitl14" in dino_model_name: dim = 1024
            elif "vitg14" in dino_model_name: dim = 1536
            else: raise ValueError(f"Nombre de modelo DINOv2 '{dino_model_name}' no reconocido.")
            encoder_channels = [dim] * 4

        elif encoder_name.lower() == 'classic':
            # <-- MODIFICADO: Pasamos el tamaño del modelo al encoder clásico
            self.encoder = UNetEncoder(n_channels=3, model_size=model_size)
            encoder_channels = self.encoder.channels
        else:
            raise ValueError(f"Encoder '{encoder_name}' no reconocido.")
            
        # El resto de la clase no necesita cambios. El decoder se adapta dinámicamente.
        self.decoder_blocks = nn.ModuleList()
        reversed_encoder_channels = list(reversed(encoder_channels))
        in_ch = reversed_encoder_channels[0]
        
        for i in range(len(reversed_encoder_channels) - 1):
            skip_ch = reversed_encoder_channels[i + 1]
            out_ch = in_ch // 2
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(in_ch, n_out_channels, kernel_size=1)
        
    def forward(self, x):
        original_size = x.shape[2:]
        skip_connections = self.encoder(x)
        reversed_skips = list(reversed(skip_connections))
        y = reversed_skips[0]
        for i, block in enumerate(self.decoder_blocks):
            skip = reversed_skips[i + 1]
            y = block(y, skip)
        y = self.final_conv(y)
        output = F.interpolate(y, size=original_size, mode='bilinear', align_corners=True)
        return output