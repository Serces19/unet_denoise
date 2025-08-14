# DeepCopycat - Manual de Usuario y Desarrollo

## 1. Configuración del Entorno

Se recomienda usar `conda` para gestionar las dependencias del proyecto.

```bash
# 1. Clona el repositorio
git clone <URL-DEL-REPOSITORIO>
cd DeepCopycat

# 2. Crea un nuevo entorno de Conda
conda create --name deepcopycat python=3.9

# 3. Activa el entorno
conda activate deepcopycat

# 4. Instala las dependencias
pip install -r requirements.txt
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
├── INFRASTRUCTURE.md    # Diagrama de la arquitectura
├── MANUAL.md            # Este manual
├── README.md            # Descripción general del proyecto
├── requirements.txt     # Dependencias de Python
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
