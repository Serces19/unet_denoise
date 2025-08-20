import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Logger:
    """
    Una clase simple para encapsular el SummaryWriter de TensorBoard.
    Crea un directorio de log único para cada ejecución del entrenamiento.
    """

    def __init__(self, base_log_dir="../runs", run_name=None):
        """
        Inicializa el Logger.

        Args:
            base_log_dir (str): Directorio base donde se guardarán los logs.
            run_name (str, opcional): Nombre personalizado para la ejecución.
                                       Si no se especifica, se genera uno con timestamp.
        """
        # Si no se proporciona run_name, se genera automáticamente
        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d-%H%M%S')

        self.log_dir = os.path.join(base_log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        print(f"📁 Los logs de TensorBoard se guardarán en: {self.log_dir}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, tag, value, step):
        """
        Registra un valor escalar (como la pérdida o la precisión).

        Args:
            tag (str): Nombre de la métrica (ej. 'Loss/train').
            value (float): Valor de la métrica.
            step (int): Paso del entrenamiento (ej. número de época).
        """
        self.writer.add_scalar(tag, value, step)

    def log_hparams(self, hparams, metrics):
        """
        Registra los hiperparámetros y las métricas finales de una ejecución.

        Args:
            hparams (dict): Diccionario con hiperparámetros (ej. {'lr': 1e-4}).
            metrics (dict): Métricas finales (ej. {'best_val_loss': 0.123}).
        """
        self.writer.add_hparams(hparams, metrics)

    def close(self):
        """
        Cierra el writer de TensorBoard. Debe llamarse al final del entrenamiento.
        """
        self.writer.close()