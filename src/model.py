import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Bloques de Construcción Reutilizables ---

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
    """Bloque decoder de la U-Net. (Tu versión original, ligeramente adaptada)."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Usamos ConvTranspose2d para un upsampling aprendido, una alternativa a Upsample + Conv
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        # Manejar posibles desajustes de tamaño por las convoluciones
        if x.shape != skip_connection.shape:
            x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)

# --- Encoders Modulares (Las "Columnas Vertebrales") ---

class UNetEncoder(nn.Module):
    """Encoder convolucional clásico de una U-Net."""
    def __init__(self, n_channels=3):
        super().__init__()
        self.channels = [64, 128, 256, 512, 1024]
        
        self.inc = DoubleConv(n_channels, self.channels[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[0], self.channels[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[1], self.channels[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[2], self.channels[3]))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(self.channels[3], self.channels[4]))
        
    def forward(self, x):
        # Devuelve una lista de las skip connections, de la más superficial a la más profunda
        s1 = self.inc(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4) # El "cuello de botella"
        return [s1, s2, s3, s4, s5]

class DinoV2Encoder(nn.Module):
    """Wrapper para el encoder DINOv2 (Tu versión original, ligeramente adaptada)."""
    def __init__(self, model_name='dinov2_vits14', fine_tune=True, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
        
        for param in self.dinov2.parameters():
            param.requires_grad = fine_tune

    def forward(self, x):
        # Extrae las feature maps y las reformatea a formato de imagen 2D (B, C, H, W)
        features = self.dinov2.get_intermediate_layers(x, n=self.n_layers, return_class_token=False)
        
        patch_size = self.dinov2.patch_embed.patch_size[0]
        h = x.shape[2] // patch_size
        w = x.shape[3] // patch_size
        
        # Devuelve una lista de skip connections, de la más superficial a la más profunda
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
    """
    def __init__(self, n_out_channels=3, encoder_name='dinov2', fine_tune_encoder=True):
        super().__init__()
        
        # --- 1. Selección del Encoder ---
        if encoder_name.lower() == 'dinov2':
            # Para dinov2_vitb14, todas las 4 capas de salida tienen 768 canales
            self.encoder = DinoV2Encoder(model_name='dinov2_vitb14', fine_tune=fine_tune_encoder, n_layers=4)
            encoder_channels=[768, 384, 192, 96]

        elif encoder_name.lower() == 'classic':
            self.encoder = UNetEncoder(n_channels=3)
            # Los canales de salida del encoder clásico
            encoder_channels = self.encoder.channels
        else:
            raise ValueError(f"Encoder '{encoder_name}' no reconocido. Elige 'dinov2' o 'classic'.")
            
        # --- 2. Creación del Decoder ---
        # El decoder se construye dinámicamente basándose en los canales del encoder elegido
        self.decoder_blocks = nn.ModuleList()
        
        # Invertimos los canales del encoder para el camino de subida
        reversed_encoder_channels = list(reversed(encoder_channels))
        
        # El bucle comienza desde el cuello de botella
        in_ch = reversed_encoder_channels[0]
        
        # Iteramos desde el segundo nivel más profundo hasta el más superficial
        for i in range(len(reversed_encoder_channels) - 1):
            skip_ch = reversed_encoder_channels[i + 1]
            out_ch = in_ch // 2
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        # --- 3. Capa de Salida Final ---
        # La salida final se reescala al tamaño original y se proyecta a los canales de salida
        self.final_conv = nn.Conv2d(in_ch, n_out_channels, kernel_size=1)
        
    def forward(self, x):
        # Guardamos las dimensiones originales para el reescalado final
        original_size = x.shape[2:]
        
        # 1. Obtener las skip connections del encoder elegido
        skip_connections = self.encoder(x)
        
        # 2. Procesar con el Decoder
        # Invertimos la lista para que sea fácil acceder a ellas en el camino de subida
        reversed_skips = list(reversed(skip_connections))
        
        # El cuello de botella (la capa más profunda) es el punto de partida
        y = reversed_skips[0]
        
        for i, block in enumerate(self.decoder_blocks):
            skip = reversed_skips[i + 1]
            y = block(y, skip)

        # 3. Producir la salida final
        y = self.final_conv(y)
        
        # Reescalar al tamaño de la imagen de entrada original
        output = F.interpolate(y, size=original_size, mode='bilinear', align_corners=True)
        
        return output

# --- Bloque de Pruebas ---
if __name__ == '__main__':
    # Asegúrate de que el tamaño de entrada sea un múltiplo del tamaño de parche de DINOv2 (14)
    input_tensor = torch.randn(2, 3, 224, 224)
    
    print("--- Probando la UNet con Encoder DINOv2 ---")
    model_dino = CopycatUNet(n_out_channels=3, encoder_name='dinov2')
    output_dino = model_dino(input_tensor)
    print("Forma de la entrada:", input_tensor.shape)
    print("Forma de la salida:", output_dino.shape)
    
    print("\n--- Probando la UNet con Encoder Clásico ---")
    model_classic = CopycatUNet(n_out_channels=1, encoder_name='classic')
    output_classic = model_classic(input_tensor)
    print("Forma de la entrada:", input_tensor.shape)
    print("Forma de la salida:", output_classic.shape)
    
    # Calcular parámetros del modelo clásico
    trainable_params = sum(p.numel() for p in model_classic.parameters() if p.requires_grad)
    print(f"Parámetros entrenables (Clásico): {trainable_params:,}")