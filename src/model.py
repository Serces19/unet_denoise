import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderBlock(nn.Module):
    """
    Bloque decoder de la U-Net.
    Realiza un upsampling, concatena con la skip connection y aplica convoluciones.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        conv_in_channels = in_channels + skip_channels
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.up(x)
        if skip_connection is not None:
            skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

class DinoV2Encoder(nn.Module):
    """
    Wrapper para el encoder DINOv2. Carga el modelo, permite el fine-tuning
    y extrae las feature maps de las capas intermedias en formato 2D.
    """
    def __init__(self, model_name='dinov2_vits14', fine_tune=True, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
        
        for param in self.dinov2.parameters():
            param.requires_grad = fine_tune

        self.dinov2.n_blocks = len(self.dinov2.blocks)

    def forward(self, x):
        features = self.dinov2.get_intermediate_layers(x, n=self.n_layers, return_class_token=False)
        
        patch_size = self.dinov2.patch_embed.patch_size[0]
        h = x.shape[2] // patch_size
        w = x.shape[3] // patch_size
        dim = features[0].shape[-1]
        
        reshaped_features = []
        for f in features:
            reshaped_f = f.permute(0, 2, 1).view(-1, dim, h, w)
            reshaped_features.append(reshaped_f)
            
        return reshaped_features

class SimpleCopyCat(nn.Module):
    """
    Modelo principal que une el Encoder DINOv2 con un Decoder tipo U-Net.
    """
    def __init__(self, n_out_channels=3, fine_tune_encoder=True):
        super().__init__()
        
        self.encoder = DinoV2Encoder(model_name='dinov2_vits14', fine_tune=fine_tune_encoder, n_layers=4)
        
        encoder_channels = 384
        # Reducimos a 3 bloques para que coincida con las skip connections disponibles
        decoder_channels = [256, 128, 64]
        
        self.decoder_blocks = nn.ModuleList()
        
        in_ch = encoder_channels
        for out_ch in decoder_channels:
            skip_ch = encoder_channels
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv2d(decoder_channels[-1], n_out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = self.encoder(x)
        skip_connections = list(reversed(skip_connections))
        
        y = skip_connections[0]
        
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[i + 1]
            y = block(y, skip)

        # Reescalado final para que la salida tenga el mismo tamaño que la entrada
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
        output = self.final_conv(y)
        
        return output

# Para probar la dimensionalidad del modelo:
if __name__ == '__main__':
    input_tensor = torch.randn(2, 3, 252, 252)
    model = SimpleCopyCat(n_out_channels=3, fine_tune_encoder=True)
    output_tensor = model(input_tensor)
    
    print("Forma de la entrada:", input_tensor.shape)
    print("Forma de la salida:", output_tensor.shape)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Parámetros totales:      {total_params:,}")