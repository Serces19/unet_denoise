# DeepCopycat - Manual de Usuario y Desarrollo

## 1. Configuración del Entorno

Usa el archivo `conda.yaml` proporcionado para crear el entorno base con todas las dependencias necesarias para DINOv2.

```bash
# 1. Clona el repositorio
git clone <URL-DEL-REPOSITORIO>
cd DeepCopycat

# 2. Crea el entorno de Conda desde el archivo
# Esto instalará PyTorch 2.0, xFormers y otras librerías base.
conda env create -f conda.yaml

# 3. Activa el entorno
conda activate dinov2
```

Opcionalmente, si se van a realizar tareas densas como la segmentación semántica, puedes crear un entorno separado que incluya dependencias extra como `mmcv`.

```bash
# (Opcional) Crea un entorno separado con dependencias extra
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

## 2. Estructura del Proyecto

```
.
├── data/                # Directorio para los datasets (no trackeado por Git)
│   ├── colorization/    # Datos para la tarea de colorización
│   └── de-aging/        # Datos para la tarea de de-aging
├── models/              # Modelos entrenados (.pth) (no trackeado por Git)
├── notebooks/           # Jupyter notebooks para experimentación y visualización
├── results/             # Imágenes y videos generados (no trackeado por Git)
├── src/                 # Código fuente principal
│   ├── model.py         # Arquitectura del modelo
│   ├── dataset.py       # Clases de Dataset de PyTorch
│   ├── train.py         # Script de entrenamiento
│   └── losses.py        # Funciones de pérdida personalizadas
├── .gitignore           # Archivos a ignorar por Git
├── conda.yaml           # Fichero de entorno para Conda
├── conda-extras.yaml    # Fichero de entorno con dependencias extra
├── INFRASTRUCTURE.md    # Diagrama de la arquitectura
├── MANUAL.md            # Este manual
├── README.md            # Descripción general del proyecto
└── TASKS.md             # Plan de desarrollo y lista de tareas
```

## 3. Cómo Ejecutar

### a) Preparar los Datos

Coloca tus imágenes en los subdirectorios correspondientes dentro de `data/`.

### b) Entrenamiento

Para iniciar el entrenamiento, ejecuta el script `train.py`. Puedes configurarlo con argumentos de línea de comandos (a ser implementados).

```bash
# Ejemplo de ejecución
python src/train.py --dataset colorization --epochs 50 --batch_size 16
```

### c) Inferencia

(A definir) Se creará un script `inference.py` para cargar un modelo entrenado y procesar una imagen o un directorio de imágenes.
