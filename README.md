# DeepCopycat - Final Project

Este proyecto implementa un modelo de traducción de imagen a imagen basado en una arquitectura U-Net con un encoder pre-entrenado DINOv2.

## Descripción

El objetivo es crear un pipeline flexible que pueda ser aplicado a diversas tareas de "copia" de un dominio de imagen a otro. El proyecto se desarrolla en dos fases principales:

1.  **Tarea de Prueba (Colorización):** Se entrena al modelo para convertir imágenes en escala de grises a imágenes a color. Esto sirve como un baseline para validar la arquitectura y el bucle de entrenamiento.
2.  **Tarea Final (De-aging):** Se aplica el mismo pipeline para rejuvenecer rostros en imágenes, utilizando un dataset creado con FaceApp.

## Componentes Clave

*   `src/model.py`: Contiene la definición de la arquitectura U-Net + DINOv2.
*   `src/dataset.py`: Gestiona la carga y pre-procesamiento de los datos.
*   `src/train.py`: Implementa el bucle de entrenamiento principal.
*   `src/losses.py`: Define las funciones de pérdida, incluyendo L1 y la Pérdida Piramidal Laplaciana.
