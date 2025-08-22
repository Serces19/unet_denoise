# Framework de Traducción Imagen-a-Imagen

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Logging-orange.svg)

**DeepCopycat** es un framework flexible y modular construido en PyTorch para tareas de traducción de imagen a imagen. El proyecto utiliza una arquitectura U-Net que puede ser potenciada por diferentes "columnas vertebrales" (encoders), permitiendo una experimentación rápida y la adaptación a diversos problemas visuales como la generación de máscaras (mattes), segmentación, o colorización.

## ✨ Características Principales

* **Arquitectura U-Net Modular:** Un diseño limpio que separa el encoder del decoder.
* **Encoders Intercambiables:** Soporta tanto un **encoder convolucional clásico** (eficiente y robusto) como un **encoder Vision Transformer (DINOv2)** pre-entrenado para un entendimiento semántico de vanguardia.
* **Monitoreo en Vivo con TensorBoard:** Visualiza las curvas de pérdida de entrenamiento y validación en tiempo real a través de una interfaz web.
* **Inferencia de Alta Resolución:** Incluye un script de inferencia con una técnica de "ventana deslizante con solapamiento" para procesar imágenes de cualquier tamaño (HD, 4K) sin artefactos ni costuras.
* **Entrenamiento Flexible:** Configuración completa a través de argumentos de línea de comandos, incluyendo la capacidad de reanudar el entrenamiento desde un checkpoint guardado.
* **Carga de Datos Robusta:** El `Dataset` personalizado maneja de forma segura el emparejamiento de archivos con nombres inconsistentes y aplica aumentación de datos de forma sincronizada.

## 📂 Estructura del Proyecto
u_proyecto/
├── data/              # Directorio para los datasets (no trackeado por Git)
│   └── mate/
│       ├── rgb/
│       └── mask/
├── models/            # Modelos entrenados (.pth) (no trackeado por Git)
├── runs/              # Logs de TensorBoard (no trackeado por Git)
└── src/               # Código fuente principal
├── utils/
│   └── logger.py  # Módulo de logging para TensorBoard
├── dataset.py     # Clases de Dataset (RedChannelDataset, etc.)
├── model.py       # Arquitectura modular CopycatUNet
├── train.py       # Script de entrenamiento principal
└── inference.py   # Script de inferencia para imágenes de alta resolución


## ⚙️ Configuración del Entorno

1.  **Clona el repositorio:**
    ```bash
    git clone <URL-DEL-REPOSITORIO>
    cd tu_proyecto
    ```
2.  **Crea el entorno de Conda:** (Si tienes un `conda.yaml`)
    ```bash
    conda env create -f conda.yaml
    conda activate dinov2
    ```
3.  **Instala las dependencias necesarias:**
    ```bash
    pip install torch torchvision numpy opencv-python tqdm pillow tensorboard
    ```

## 🚀 Guía de Uso

Todos los comandos se ejecutan desde la carpeta `src/`.

### 1. Preparar el Dataset

Asegúrate de que tus datos sigan la estructura esperada dentro de la carpeta `data/`. Por ejemplo, para la tarea de generación de mattes:
* Imágenes de entrada en `../data/mate/rgb/`
* Máscaras objetivo en `../data/mate/mask/`

### 2. Entrenamiento

El script `train.py` es altamente configurable.

**Comando de Entrenamiento Básico (con encoder clásico):**
Este comando inicia un entrenamiento de 150 épocas con la UNet clásica, usando la pérdida BCE recomendada para máscaras.

```bash
python train.py --encoder classic --loss_fn bce --num_epochs 150 --batch_size 8
```
Entrenamiento con el Encoder DINOv2:
Aquí usamos el encoder DINOv2. Nota que --crop_size debe ser un múltiplo de 14 (el tamaño de parche de DINOv2).


```bash

python train.py --encoder dinov2 --loss_fn bce --num_epochs 100 --crop_size 392 --batch_size 4
```
Reanudar un Entrenamiento Interrumpido:
Si un entrenamiento se detuvo, puedes continuarlo desde el último checkpoint guardado usando el flag --resume_from. El script cargará el modelo, el optimizador y continuará desde la época correcta.

```bash
# Supongamos que el entrenamiento anterior llegó a 100 épocas y quieres continuar hasta 250
python train.py --encoder classic --num_epochs 250 --resume_from ../models/best_model_classic.pth
```

### 3. Monitoreo con TensorBoard
Mientras tu modelo entrena, puedes visualizar las gráficas de pérdida en vivo.

Inicia el entrenamiento en una terminal como se describió arriba.

Abre una SEGUNDA terminal.

Navega a la raíz de tu proyecto (la carpeta que contiene src/, runs/, etc.).

Ejecuta el servidor de TensorBoard:

```bash
tensorboard --logdir runs
```
Abre tu navegador web y ve a la dirección que te indica la terminal (usualmente http://localhost:6006/). Verás tus gráficas actualizándose después de cada época.

### 4. Inferencia
Una vez que tengas un modelo entrenado (.pth), usa inference.py para procesar imágenes nuevas en alta resolución.

Inferencia con un Modelo "Clásico":
Usa el archivo .pth generado con el encoder classic y especifica --encoder classic.

```bash
python inference.py \
    --model_path "../models/best_model_classic.pth" \
    --input "../data/imagenes_para_probar/foto_01.png" \
    --output "../outputs/foto_01_matte.png" \
    --encoder classic \
    --n_out_channels 1 \
    --window_size 1024
```

Inferencia con un Modelo "DINOv2":
Apunta al modelo entrenado con dinov2 y especifica --encoder dinov2. El script ajustará automáticamente el --window_size si no es un múltiplo de 14.

```bash
python inference.py \
    --model_path "../models/best_model_dinov2.pth" \
    --input "../data/imagenes_para_probar/foto_02.png" \
    --output "../outputs/foto_02_matte.png" \
    --encoder dinov2 \
    --n_out_channels 1 \
    --window_s
```

