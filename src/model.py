import torch
import torch.nn as nn
import math

class DecoderBlock(nn.Module):
    """
    Bloque decoder de la U-Net.
    Realiza un upsampling, concatena con la skip connection y aplica convoluciones.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # La entrada a la convolución será la suma de los canales del paso anterior y de la skip connection
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
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Congelar o no los parámetros para fine-tuning
        for param in self.dinov2.parameters():
            param.requires_grad = fine_tune

        # El método get_intermediate_layers de DINOv2 es perfecto para extraer skip connections
        # Extraemos las últimas n_layers capas. Para ViT-S (12 bloques), serían las capas 8, 9, 10, 11
        self.dinov2.n_blocks = len(self.dinov2.blocks)
        self.layer_indices = [self.dinov2.n_blocks - n_layers + i for i in range(n_layers)]


    def forward(self, x):
        # DINOv2 espera imágenes de un tamaño específico, ej. 224x224.
        # El pre-procesamiento en el dataset se encargará de esto.
        
        # Extraemos los patch embeddings de las capas intermedias
        # El resultado es una lista de tensores de forma [Batch, NumPatches, Dim]
        features = self.dinov2.get_intermediate_layers(x, n=self.n_layers, return_class_token=False)
        
        # --- El paso CRÍTICO: Remodelar de Patches a 2D Feature Maps ---
        # La salida de DINOv2 no es espacial, debemos reconstruirla.
        # Para una imagen de entrada (H, W) y un patch_size (P), la feature map será (H/P, W/P)
        patch_size = self.dinov2.patch_embed.patch_size[0]
        h = x.shape[2] // patch_size
        w = x.shape[3] // patch_size
        
        # Dimensión de los embeddings en ViT-S es 384
        dim = features[0].shape[-1]
        
        reshaped_features = []
        for f in features:
            # [B, NumPatches, Dim] -> [B, Dim, H, W]
            reshaped_f = f.permute(0, 2, 1).view(-1, dim, h, w)
            reshaped_features.append(reshaped_f)
            
        return reshaped_features


class SimpleCopyCat(nn.Module):
    """
    Modelo principal que une el Encoder DINOv2 con un Decoder tipo U-Net.
    """
    def __init__(self, n_out_channels=3, fine_tune_encoder=True):
        super().__init__()
        
        # --- Encoder ---
        # Usamos 4 capas para las skip connections
        self.encoder = DinoV2Encoder(model_name='dinov2_vits14', fine_tune=fine_tune_encoder, n_layers=4)
        
        # --- Decoder ---
        # Las dimensiones de los canales dependen del encoder (ViT-S -> 384)
        encoder_channels = 384
        decoder_channels = [256, 128, 64, 32]
        
        # Las skip connections vienen en orden: de la más superficial a la más profunda.
        # Las procesaremos en orden inverso.
        self.decoder_blocks = nn.ModuleList()
        
        # El primer bloque del decoder toma la salida más profunda del encoder
        in_ch = encoder_channels
        for out_ch in decoder_channels:
            skip_ch = encoder_channels # Todas las skips de DINOv2 tienen la misma dimensión
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # --- Capa Final ---
        # El decoder nos deja en una resolución de 224x224 (14 * 2^4 = 224)
        # con `decoder_channels[-1]` canales.
        # La última capa de upsampling y la convolución final
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(decoder_channels[-1], n_out_channels, kernel_size=1)
        
    def forward(self, x):
        # Obtenemos las skip connections del encoder.
        # Lista de tensores, ej: [capa_8, capa_9, capa_10, capa_11]
        skip_connections = self.encoder(x)
        
        # Las invertimos para empezar el decoding desde la más profunda
        skip_connections = list(reversed(skip_connections))
        
        # La primera entrada al decoder es la feature map más profunda
        y = skip_connections[0]
        
        # Bucle del decoder
        for i, block in enumerate(self.decoder_blocks):
            # Las siguientes skip connections son las de las capas menos profundas
            skip = skip_connections[i+1] if i + 1 < len(skip_connections) else None
            
            # En la U-Net clásica, la skip connection debe tener el mismo tamaño
            # que la feature map después del upsampling. Aquí, las skips de DINOv2
            # tienen todas el mismo tamaño (ej. 16x16), lo cual es una simplificación.
            # Para una implementación más avanzada, se podrían hacer upsample de las skips.
            # Por ahora, las usamos directamente.
            if skip is not None:
                 y = block(y, skip)
            else:
                # Podríamos tener un último bloque que no usa skip connection
                pass

        # Upsampling final para recuperar tamaño original si es necesario
        # Con patch_size=14, la feature map es 16x16. 4 bloques decoder -> 16*(2^4) = 256
        # Esto es cercano a 224, así que no necesitamos un upsample extra aquí.
        
        # Convolución final para generar la imagen de salida
        output = self.final_conv(y)
        
        return output

# Para probar la dimensionalidad del modelo:
if __name__ == '__main__':
    # Imagen de entrada de ejemplo (Batch, Canales, Altura, Ancho)
    # DINOv2 fue entrenado con imágenes de 224x224
    input_tensor = torch.randn(2, 3, 224, 224)
    
    # Crear el modelo
    model = SimpleCopyCat(n_out_channels=3, fine_tune_encoder=True)
    
    # Pasar la imagen por el modelo
    output_tensor = model(input_tensor)
    
    # Comprobar la forma de la salida
    print("Forma de la entrada:", input_tensor.shape)
    print("Forma de la salida:", output_tensor.shape)
    
    # Contar parámetros entrenables
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Parámetros totales:      {total_params:,}")

