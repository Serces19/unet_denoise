# Project Tasks - DeepCopycat

## Semana 1: Arquitectura del Modelo y Datos de Prueba (Ahora - Vie. 15 de Agosto)
### 🎯 Objetivo Principal: 
Tener la arquitectura completa del modelo U-Net+DINOv2 codificada y un generador de datos funcional para nuestra tarea de prueba.

#### 📋 Tareas Específicas:
- [ ] **Setup del Proyecto:**
    - [x] Crear la estructura de carpetas (/src, /data, /models, etc.).
    - [x] Inicializar un repositorio de Git.
    - [ ] Configurar el entorno de Conda e instalar las librerías base (torch, torchvision, tqdm, opencv-python, scikit-image).
- [ ] **Implementación del Encoder (model.py):**
    - [ ] Investigar el repositorio oficial de DINOv2 de Meta.
    - [ ] Escribir una clase `DinoV2Encoder(nn.Module)` que cargue el modelo DINOv2 pre-entrenado.
    - [ ] Modificar su método `forward` para que devuelva las activaciones de capas intermedias (skip-connections).
- [ ] **Implementación del Decoder (model.py):**
    - [ ] Crear los bloques de `Upsample + Conv2D` del decodificador.
    - [ ] Ensamblar el modelo final `SimpleCopyCat(nn.Module)` conectando el encoder y el decodificador.
- [ ] **Generador de Datos de Prueba (dataset.py):**
    - [ ] Crear una clase `ColorizationDataset(Dataset)` de PyTorch.
    - [ ] Implementar `__getitem__` para cargar una imagen, convertirla a escala de grises (input) y mantener la original (target).
    - [ ] Aplicar las transformaciones necesarias (resize, normalización).

#### ✅ Entregable Clave al final de la Semana 1:
- [ ] Un archivo `model.py` con la arquitectura U-Net+DINOv2.
- [ ] Un archivo `dataset.py` capaz de generar pares de (gris, color).

---

## Semana 2: El Bucle de Entrenamiento y el Modelo Baseline (Lun. 18 - Vie. 22 de Agosto)
### 🎯 Objetivo Principal: 
Escribir, depurar y ejecutar con éxito el script de entrenamiento para obtener un primer modelo funcional (baseline).

#### 📋 Tareas Específicas:
- [ ] **Implementación del Bucle de Entrenamiento (train.py):**
    - [ ] Carga del modelo y del `DataLoader`.
    - [ ] Definición del optimizador (ej. `torch.optim.AdamW`).
    - [ ] Bucle principal de épocas y lotes.
    - [ ] Lógica para mover tensores a la GPU (`.to(device)`).
- [ ] **Selección de Pérdida Simple:**
    - [ ] Instanciar y usar `nn.L1Loss()`.
- [ ] **Integración y Debugging:**
    - [ ] Conectar `train.py`, `model.py` y `dataset.py`.
    - [ ] Resolver errores de dimensiones, tipos y memoria.
- [ ] **Lanzamiento del Primer Entrenamiento:**
    - [ ] Configurar instancia en la nube (AWS).
    - [ ] Ejecutar `train.py` y guardar los pesos del modelo.

#### ✅ Entregable Clave al final de la Semana 2:
- [ ] Un script `train.py` funcional.
- [ ] Pesos del modelo: `baseline_colorizer.pth`.
- [ ] Imagen de muestra con la predicción del baseline.

---

## Semana 3: Evaluación, Mejora y la Pérdida Avanzada (Lun. 25 - Vie. 29 de Agosto)
### 🎯 Objetivo Principal: 
Analizar el rendimiento del baseline y mejorarlo con una función de pérdida avanzada.

#### 📋 Tareas Específicas:
- [ ] **Evaluación Cualitativa del Baseline:**
    - [ ] Analizar visualmente los resultados de la colorización.
- [ ] **Implementación de la Pérdida Laplaciana Piramidal (losses.py):**
    - [ ] Crear el archivo `losses.py`.
    - [ ] Implementar una función que genere la pirámide Laplaciana de una imagen.
    - [ ] Crear la clase `PyramidL1Loss(nn.Module)` que calcule la pérdida L1 ponderada en cada nivel de la pirámide.
- [ ] **Fine-tuning con Pérdida Avanzada:**
    - [ ] Modificar `train.py` para usar la nueva pérdida.
    - [ ] Cargar `baseline_colorizer.pth` y continuar el entrenamiento.

#### ✅ Entregable Clave al final de la Semana 3:
- [ ] El script `losses.py` con la pérdida piramidal.
- [ ] Pesos del modelo mejorado: `advanced_colorizer.pth`.
- [ ] Comparativa: Original | Resultado L1 | Resultado L1+Piramidal.

---

## Semana 4: Aplicación a "De-aging" y Documentación Final (Lun. 1 - Vie. 5 de Septiembre)
### 🎯 Objetivo Principal: 
Aplicar el pipeline a la tarea de De-aging y preparar los entregables finales.

#### 📋 Tareas Específicas:
- [ ] **Generación del Dataset De-aging:**
    - [ ] Procesar imágenes con FaceApp para crear el dataset (original, envejecida).
- [ ] **Entrenamiento del Modelo De-aging:**
    - [ ] Modificar `dataset.py` para el nuevo dataset.
    - [ ] Re-ejecutar `train.py` para entrenar el modelo `SimpleDe-age`.
- [ ] **Análisis Final de Resultados:**
    - [ ] Evaluar cualitativamente los resultados del modelo de envejecimiento.
- [ ] **Creación de Entregables Finales:**
    - [ ] Escribir el informe técnico final.
    - [ ] Producir el video de demostración.
    - [ ] Limpiar, comentar y empaquetar el repositorio de Git.

#### ✅ Entregable Clave al final de la Semana 4:
- [ ] Informe Técnico Final.
- [ ] Video de Demostración.
- [ ] Repositorio de Git completo.
