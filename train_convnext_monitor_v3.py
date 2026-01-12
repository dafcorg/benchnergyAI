import os  # Módulo estándar para manejar rutas, directorios y operaciones de sistema de archivos
import time  # Proporciona funciones relacionadas con el tiempo (timestamps, sleep, etc.)
import threading  # Permite crear y gestionar hilos de ejecución (lo usamos para el monitor de la GPU)
import math  # Funciones matemáticas adicionales (sqrt, log, etc.)
import csv  # Para leer y escribir archivos CSV
import json  # Para leer y escribir archivos JSON (config y resumen de experimento)
import argparse  # Para parsear argumentos de línea de comandos
import random  # Generador de números aleatorios de la librería estándar
import datetime  # Manejo de fechas y horas (timestamps legibles)
from dataclasses import dataclass, asdict  # dataclass para estructuras de datos y asdict para convertirlas a diccionarios
from typing import Optional, List, Dict, Any  # Tipos auxiliares para anotaciones de tipo
from contextlib import nullcontext  # Contexto vacío útil cuando no queremos usar autocast

import numpy as np  # Librería de cálculo numérico para arrays y operaciones matemáticas
import torch  # Framework principal de deep learning
import torch.nn as nn  # Módulo de redes neuronales (capas, pérdidas, etc.)
import torch.optim as optim  # Optimizadores (AdamW, SGD, etc.)
from torch.utils.data import DataLoader  # Cargador de datos por lotes
from torchvision import datasets, transforms, models  # Datasets, transformaciones y modelos preentrenados

import pynvml  # Envoltorio de la NVML de NVIDIA para monitorizar GPU (potencia, memoria, temperatura)
import matplotlib.pyplot as plt  # Librería para generar gráficos
from sklearn.metrics import confusion_matrix, f1_score  # Métricas de evaluación: matriz de confusión y F1
from torch.cuda.amp import autocast, GradScaler  # Herramientas de AMP (automatic mixed precision) para FP16/BF16


# =========================
# Configuración por defecto
# =========================

CONFIG_DEFAULT = {
    "data_dir": "./imagenet1k",     # Ruta por defecto donde se espera el dataset con carpetas train/ y val/
    "log_dir": "./logs_convnext",   # Directorio base donde se guardarán los resultados de los experimentos
    "batch_size": 128,              # Tamaño de lote por defecto para entrenamiento y validación
    "num_workers": 8,               # Número de procesos/hilos para cargar los datos en paralelo
    "num_epochs": 20,               # Número de épocas por defecto para el entrenamiento
    "lr": 1e-3,                     # Tasa de aprendizaje inicial por defecto
    "weight_decay": 0.05,           # Parámetro de regularización L2 (weight decay) para AdamW
    "monitor_interval": 1.0,        # Intervalo en segundos entre muestras del monitor de GPU
    "seed": 1234,                   # Semilla para garantizar reproducibilidad
    "device": "cuda:0",             # Dispositivo por defecto (primera GPU). Si no hay GPU, se cae a CPU más adelante
    "precision": "fp32",            # Precisión numérica por defecto: float32 (sin AMP)
    "enable_sm_metrics": False,     # Flag para activar métricas SM embebidas (actualmente placeholder)
    "ncu_csv_path": None,           # Ruta al CSV generado por Nsight Compute; se usa para sm_profile
}

# =========================
# Parsing de argumentos CLI
# =========================

def parse_args() -> Dict[str, Any]:
    """Parsa los argumentos de línea de comandos y devuelve un diccionario de configuración."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento ConvNeXt con monitorización y tracking JSON"  # Descripción que aparece en --help
    )
    parser.add_argument("--data_dir", type=str, default=CONFIG_DEFAULT["data_dir"])  # Ruta al dataset
    parser.add_argument("--log_dir", type=str, default=CONFIG_DEFAULT["log_dir"])  # Ruta base para logs/experimentos
    parser.add_argument("--batch_size", type=int, default=CONFIG_DEFAULT["batch_size"])  # Tamaño de batch
    parser.add_argument("--num_workers", type=int, default=CONFIG_DEFAULT["num_workers"])  # Workers de DataLoader
    parser.add_argument("--epochs", type=int, default=CONFIG_DEFAULT["num_epochs"])  # Número de épocas
    parser.add_argument("--lr", type=float, default=CONFIG_DEFAULT["lr"])  # Tasa de aprendizaje
    parser.add_argument("--weight_decay", type=float, default=CONFIG_DEFAULT["weight_decay"])  # Weight decay
    parser.add_argument("--monitor_interval", type=float, default=CONFIG_DEFAULT["monitor_interval"])  # Intervalo monitor
    parser.add_argument("--seed", type=int, default=CONFIG_DEFAULT["seed"])  # Semilla para reproducibilidad
    parser.add_argument("--device", type=str, default=CONFIG_DEFAULT["device"])  # Dispositivo (ej. "cuda:0" o "cpu")
    parser.add_argument(
        "--precision",
        type=str,
        default=CONFIG_DEFAULT["precision"],        # Valor por defecto de precisión (fp32)
        choices=["fp32", "fp16", "bf16"],           # Restricción a tres opciones válidas
        help="Precisión numérica: fp32, fp16 o bf16",  # Mensaje explicativo en --help
    )
    parser.add_argument(
        "--enable_sm_metrics",
        action="store_true",                        # Flag booleano; si se pasa, se pone a True
        help="Placeholder para CUPTI embebido (no hace nada aún).",  # Descripción del flag
    )
    parser.add_argument(
        "--ncu_csv_path",
        type=str,
        default=CONFIG_DEFAULT["ncu_csv_path"],     # Por defecto None si no se especifica
        help="Ruta al CSV generado por Nsight Compute (ncu --csv --log-file ...).",  # Ayuda
    )

    args = parser.parse_args()  # Parsea los argumentos reales de la línea de comandos
    config = {
        "data_dir": args.data_dir,                  # Propaga argumento data_dir al diccionario config
        "log_dir": args.log_dir,                    # Propaga log_dir
        "batch_size": args.batch_size,              # Propaga batch_size
        "num_workers": args.num_workers,            # Propaga num_workers
        "num_epochs": args.epochs,                  # Propaga epochs (renombrado a num_epochs)
        "lr": args.lr,                              # Propaga learning rate
        "weight_decay": args.weight_decay,          # Propaga weight_decay
        "monitor_interval": args.monitor_interval,  # Propaga intervalo de monitorización
        "seed": args.seed,                          # Propaga semilla
        "device": args.device,                      # Propaga dispositivo
        "precision": args.precision,                # Propaga precisión numérica
        "enable_sm_metrics": args.enable_sm_metrics,  # Propaga flag de métricas SM embebidas
        "ncu_csv_path": args.ncu_csv_path,          # Propaga ruta al CSV de Nsight Compute
    }
    return config  # Devuelve el diccionario de configuración para el resto del script



# =========================
# Semilla global
# =========================

def set_seed(seed: int):
    """Configura las semillas de distintos generadores aleatorios para reproducibilidad."""
    random.seed(seed)                    # Semilla para el generador aleatorio de la librería estándar
    np.random.seed(seed)                 # Semilla para NumPy
    torch.manual_seed(seed)              # Semilla para el generador aleatorio de PyTorch en CPU
    torch.cuda.manual_seed_all(seed)     # Semilla para todos los generadores aleatorios en GPUs
    torch.backends.cudnn.deterministic = True  # Fuerza operaciones deterministas en CUDNN (puede ralentizar un poco)
    torch.backends.cudnn.benchmark = False     # Desactiva heurísticas de CUDNN que podrían introducir no determinismo


# =========================
# Estado compartido
# =========================

@dataclass
class AlgoState:
    """Representa el estado actual del algoritmo de entrenamiento/validación."""
    timestamp: float = 0.0          # Marca de tiempo del último update de estado (segundos desde epoch)
    epoch: int = 0                  # Número de época actual
    global_step: int = 0            # Contador global de pasos (batches vistos)
    batch_in_epoch: int = 0         # Índice de batch dentro de la época actual
    phase: str = "idle"             # Fase actual: "train", "val" o "idle"
    batch_loss: float = math.nan    # Última pérdida de batch
    batch_acc: float = math.nan     # Última accuracy de batch (en %)
    running_loss: float = math.nan  # Pérdida acumulada/promedio hasta el batch actual
    running_acc: float = math.nan   # Accuracy acumulada/promedio hasta el batch actual
    grad_norm: float = math.nan     # Norma L2 del gradiente (para diagnosticar estabilidad)
    lr: float = math.nan            # Tasa de aprendizaje actual (primer grupo del optimizador)



@dataclass
class MonitorRecord:
    """Registro completo de una muestra del monitor de GPU más el estado del algoritmo."""
    timestamp: float                # Tiempo absoluto de la muestra (segundos desde epoch)
    rel_time_s: float               # Tiempo relativo desde el inicio del experimento
    power_W: float                  # Potencia instantánea de la GPU en vatios
    util_gpu: float                 # Utilización de la GPU en porcentaje (0–100)
    util_mem: float                 # Utilización de la controladora de memoria en porcentaje
    mem_used_MB: float              # Memoria de la GPU utilizada en megabytes
    temp_C: float                   # Temperatura de la GPU en grados Celsius

    epoch: int                      # Época actual del entrenamiento en el momento de la muestra
    global_step: int                # Paso global (batch total) en el momento de la muestra
    batch_in_epoch: int             # Índice del batch dentro de la época cuando se tomó la muestra
    phase: str                      # Fase del algoritmo: "train", "val" o "idle"
    batch_loss: float               # Última pérdida de batch conocida
    batch_acc: float                # Última accuracy de batch conocida (%)
    running_loss: float             # Pérdida media acumulada hasta ese punto
    running_acc: float              # Accuracy media acumulada hasta ese punto
    grad_norm: float                # Norma L2 del gradiente en el último paso
    lr: float                       # Tasa de aprendizaje en el último paso

    # Métricas SM avanzadas (se rellenan solo si se implementa _collect_sm_metrics)
    sm_occupancy: Optional[float] = None   # Ocupación de SM (porcentaje u otro indicador)
    active_warps: Optional[float] = None   # Número de warps activos (o media)
    inst_executed: Optional[float] = None  # Número de instrucciones ejecutadas (indicador agregado)
    stalls: Optional[float] = None         # Métrica de stalls (por ejemplo, porcentaje o contador agregado)



class SharedState:
    """Gestiona el estado compartido entre el hilo de entrenamiento y el hilo de monitorización."""
    def __init__(self):
        self._lock = threading.Lock()      # Lock para proteger el acceso concurrente al estado
        self._algo_state = AlgoState()     # Estado del algoritmo inicializado con valores por defecto
        self._start_time = time.time()     # Tiempo de inicio del experimento (para tiempos relativos)

    def update_algo_state(self, **kwargs):
        """
        Actualiza campos específicos del estado del algoritmo.
        Se llama desde el loop de entrenamiento/validación.
        """
        with self._lock:                   # Asegura exclusión mutua entre hilos
            for k, v in kwargs.items():    # Recorre los pares clave-valor recibidos
                if hasattr(self._algo_state, k):  # Solo actualiza si el atributo existe en AlgoState
                    setattr(self._algo_state, k, v)  # Asigna el nuevo valor al atributo correspondiente
            self._algo_state.timestamp = time.time()  # Actualiza el timestamp del último update

    def snapshot(self) -> (AlgoState, float):
        """
        Devuelve una copia del estado del algoritmo y el tiempo relativo desde el inicio.
        Se usa desde el hilo de monitorización para leer un estado consistente.
        """
        with self._lock:                         # Bloquea mientras se crea la copia
            algo = AlgoState(**asdict(self._algo_state))  # Crea un nuevo AlgoState copiando campos del actual
        rel_t = algo.timestamp - self._start_time         # Calcula el tiempo relativo respecto al inicio
        return algo, rel_t                                # Devuelve el estado copiado y el tiempo relativo

    @property
    def start_time(self):
        """Devuelve el tiempo de inicio del experimento."""
        return self._start_time


shared_state = SharedState()  # Instancia global de SharedState para que training y monitor compartan información



# =========================
# Monitor GPU + estado
# =========================

class GPUMonitorThread(threading.Thread):
    """Hilo que monitoriza la GPU periódicamente y captura también el estado del algoritmo."""
    def __init__(self, device_index=0, interval=1.0, enable_sm_metrics=False):
        super().__init__()                      # Inicializa la clase base Thread
        self.device_index = device_index        # Índice de la GPU a monitorizar (0 por defecto)
        self.interval = interval                # Intervalo en segundos entre muestras (ej. 1.0 s)
        self.enable_sm_metrics = enable_sm_metrics  # Flag para activar métricas SM embebidas (placeholder)
        self._stop_flag = threading.Event()     # Evento para señalizar parada del hilo
        self.records: List[MonitorRecord] = []  # Lista donde se irán acumulando los registros MonitorRecord

        pynvml.nvmlInit()                       # Inicializa la NVML de NVIDIA
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)  # Obtiene un handle a la GPU seleccionada


    def _collect_sm_metrics(self):
        """
        Punto de extensión para integrar CUPTI / Nsight embebido.
        De momento devuelve None en todo para no añadir overhead extra.
        """
        return None, None, None, None          # Devuelve cuatro valores None (sin métricas SM reales)

    def run(self):
        """Método principal del hilo; se ejecuta en bucle hasta que se pida la parada."""
        while not self._stop_flag.is_set():    # Bucle mientras no se haya activado el flag de parada
            now = time.time()                  # Timestamp actual en segundos
            algo_state, rel_t = shared_state.snapshot()  # Captura el estado del algoritmo y el tiempo relativo


            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)  # Lee potencia instantánea en miliwatios
            power_W = power_mW / 1000.0                             # Convierte la potencia a vatios
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)  # Lee utilización de GPU y memoria (%)
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU             # Lee la temperatura de la GPU (ºC)
            )
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)   # Lee información de memoria (usada, total, libre)
            mem_used_MB = mem_info.used / (1024 ** 2)                # Convierte memoria usada de bytes a megabytes


            if self.enable_sm_metrics:
                # Si el flag está activo, se llama al gancho para (en el futuro) obtener métricas SM reales
                sm_occupancy, active_warps, inst_executed, stalls = self._collect_sm_metrics()
            else:
                # Si no está activo, se dejan las métricas SM como None para no introducir overhead extra
                sm_occupancy = None
                active_warps = None
                inst_executed = None
                stalls = None


            rec = MonitorRecord(
                timestamp=now,                    # Tiempo absoluto de la muestra
                rel_time_s=rel_t,                 # Tiempo relativo desde inicio del experimento
                power_W=power_W,                  # Potencia en vatios
                util_gpu=util.gpu,                # Utilización de la GPU en %
                util_mem=util.memory,             # Utilización de la controladora de memoria en %
                mem_used_MB=mem_used_MB,          # Memoria usada en MB
                temp_C=temp,                      # Temperatura en ºC
                epoch=algo_state.epoch,           # Época actual del algoritmo
                global_step=algo_state.global_step,      # Paso global
                batch_in_epoch=algo_state.batch_in_epoch,  # Índice de batch dentro de la época
                phase=algo_state.phase,           # Fase: "train", "val" o "idle"
                batch_loss=algo_state.batch_loss, # Última pérdida de batch
                batch_acc=algo_state.batch_acc,   # Última accuracy de batch
                running_loss=algo_state.running_loss,  # Pérdida acumulada
                running_acc=algo_state.running_acc,    # Accuracy acumulada
                grad_norm=algo_state.grad_norm,   # Norma del gradiente
                lr=algo_state.lr,                 # Learning rate
                sm_occupancy=sm_occupancy,        # Métrica SM de ocupación (placeholder)
                active_warps=active_warps,        # Warps activos (placeholder)
                inst_executed=inst_executed,      # Instrucciones ejecutadas (placeholder)
                stalls=stalls,                    # Stalls (placeholder)
            )
            self.records.append(rec)              # Añade el registro a la lista de muestras

            time.sleep(self.interval)             # Espera el intervalo configurado antes de la siguiente muestra


    def stop(self):
        """Solicita la parada del hilo, espera a que termine y libera NVML."""
        self._stop_flag.set()                    # Activa el flag de parada
        self.join()                              # Bloquea hasta que el hilo termine su ejecución
        pynvml.nvmlShutdown()                    # Cierra la sesión de NVML


    def total_energy_joules(self) -> float:
        """
        Estima la energía total consumida por la GPU (en julios)
        usando integración trapezoidal sobre las muestras de potencia.
        """
        if len(self.records) < 2:                # Si hay menos de dos muestras, no se puede integrar
            return 0.0                           # Devuelve 0.0 julios
        E = 0.0                                  # Acumulador de energía
        for i in range(len(self.records) - 1):   # Recorre pares consecutivos de muestras
            p1 = self.records[i].power_W         # Potencia en la muestra i
            p2 = self.records[i + 1].power_W     # Potencia en la muestra i+1
            t1 = self.records[i].timestamp       # Tiempo en la muestra i
            t2 = self.records[i + 1].timestamp   # Tiempo en la muestra i+1
            E += 0.5 * (p1 + p2) * (t2 - t1)     # Área del trapecio: media de potencias * delta de tiempo
        return E                                 # Devuelve energía aproximada en julios


    def export_csv(self, path: str):
        """Exporta todos los registros del monitor a un CSV en la ruta indicada."""
        if not self.records:                     # Si no hay registros, no hace nada
            return
        fieldnames = list(asdict(self.records[0]).keys())  # Usa las claves del primer registro como cabecera CSV
        with open(path, "w", newline="") as f:   # Abre el archivo en modo escritura
            writer = csv.DictWriter(f, fieldnames=fieldnames)  # Crea un escritor de CSV basado en diccionarios
            writer.writeheader()                 # Escribe la fila de cabecera
            for rec in self.records:             # Recorre todos los registros
                writer.writerow(asdict(rec))     # Convierte el dataclass a dict y lo escribe como fila



# =========================
# Datos y modelo
# =========================

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    """
    Crea DataLoaders de entrenamiento y validación a partir de un directorio
    con la siguiente estructura:
      data_dir/
        train/
          clase_0/
          ...
        val/
          clase_0/
          ...
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],    # Media por canal usada típicamente en ImageNet (RGB)
        std=[0.229, 0.224, 0.225],     # Desviaciones estándar por canal para normalización
    )

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),   # Recorta aleatoriamente y redimensiona a 224x224
        transforms.RandomHorizontalFlip(),   # Aplica flips horizontales aleatorios como aumento de datos
        transforms.ToTensor(),               # Convierte la imagen PIL/NumPy a tensor PyTorch (C,H,W en [0,1])
        normalize,                           # Aplica normalización canal a canal
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),              # Redimensiona el lado corto a 256 píxeles
        transforms.CenterCrop(224),          # Recorta centrado a 224x224
        transforms.ToTensor(),               # Convierte a tensor
        normalize,                           # Aplica la misma normalización
    ])


    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    # Crea un dataset de entrenamiento a partir de data_dir/train usando las transformaciones de train

    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)
    # Crea un dataset de validación a partir de data_dir/val con las transformaciones de validación


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,   # Mezcla los datos en cada época
        num_workers=num_workers, pin_memory=True         # Usa varios workers y fija memoria para transferencias más rápidas a GPU
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,    # No hace falta mezclar en validación
        num_workers=num_workers, pin_memory=True         # Configuración similar
    )
    num_classes = len(train_ds.classes)                  # Número de clases deducido de las subcarpetas
    return train_loader, val_loader, num_classes         # Devuelve los dataloaders y el número de clases



def get_model(num_classes: int):
    """
    Carga un ConvNeXt Tiny preentrenado en ImageNet y adapta la última capa
    al número de clases del dataset actual.
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    # Carga ConvNeXt Tiny con pesos preentrenados en ImageNet (1K clases)

    in_features = model.classifier[2].in_features
    # Obtiene el número de características de entrada de la última capa lineal

    model.classifier[2] = nn.Linear(in_features, num_classes)
    # Reemplaza la última capa por una nueva capa lineal con num_classes salidas

    return model


# =========================
# Entrenamiento
# =========================

def compute_grad_norm(model: nn.Module) -> float:
    """
    Calcula la norma L2 del gradiente de todos los parámetros del modelo.
    Útil para monitorizar estabilidad numérica y posibles explosiones de gradiente.
    """
    total_norm_sq = 0.0                              # Acumulador del cuadrado de la norma
    for p in model.parameters():                     # Recorre todos los parámetros entrenables del modelo
        if p.grad is not None:                       # Solo considera parámetros que tengan gradiente calculado
            param_norm = p.grad.data.norm(2)         # Calcula la norma L2 del gradiente de ese parámetro
            total_norm_sq += param_norm.item() ** 2  # Suma el cuadrado de esa norma al acumulador
    return math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0
    # Devuelve la raíz cuadrada del acumulador, o 0.0 si no hubo gradientes



def train_one_epoch(
    model, loader, criterion, optimizer,
    epoch, global_step, device,
    use_amp, amp_dtype, scaler,
):
    """
    Ejecuta una época completa de entrenamiento.
    Actualiza el estado compartido para el monitor y devuelve estadísticas de la época.
    """
    model.train()                                   # Pone el modelo en modo entrenamiento (activa dropout, BN en modo train)
    shared_state.update_algo_state(phase="train", epoch=epoch)
    # Informa al estado compartido de que estamos en fase de entrenamiento y de la época actual

    running_loss = 0.0                              # Acumulador de pérdida total (ponderada por tamaño de batch)
    running_correct = 0                             # Contador de aciertos acumulados
    total = 0                                       # Contador de muestras procesadas

    for batch_idx, (images, targets) in enumerate(loader):
        # Recorre todos los batches del DataLoader de entrenamiento
        images = images.to(device, non_blocking=True)    # Mueve las imágenes al dispositivo (GPU/CPU)
        targets = targets.to(device, non_blocking=True)  # Mueve las etiquetas al mismo dispositivo

        optimizer.zero_grad()                       # Resetea los gradientes acumulados del optimizador

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
            # Si estamos en GPU y se ha activado AMP, usamos autocast con el dtype configurado (fp16/bf16)
        else:
            ctx = nullcontext()
            # Si no, usamos un contexto vacío que no hace nada especial

        with ctx:
            outputs = model(images)                 # Forward pass: obtiene las predicciones del modelo
            loss = criterion(outputs, targets)      # Calcula la pérdida de clasificación (CrossEntropyLoss)

        if scaler is not None and scaler.is_enabled():
            # Si estamos usando GradScaler (típicamente en FP16)
            scaler.scale(loss).backward()           # Escala la pérdida y realiza backpropagation en precisión mixta
            scaler.unscale_(optimizer)              # Desescala los gradientes para poder medir la norma correctamente
            grad_norm = compute_grad_norm(model)    # Calcula la norma del gradiente ya desescalado
            scaler.step(optimizer)                  # Aplica el paso del optimizador con los gradientes escalados
            scaler.update()                         # Actualiza el factor de escala interno del GradScaler
        else:
            # Camino estándar: FP32 o BF16 sin escalado
            loss.backward()                         # Backpropagation normal
            grad_norm = compute_grad_norm(model)    # Calcula la norma del gradiente
            optimizer.step()                        # Actualiza los parámetros del modelo

        _, preds = outputs.max(1)                   # Toma el índice de clase con mayor probabilidad por muestra
        batch_size = targets.size(0)                # Tamaño real del batch (por si el último es más pequeño)
        total += batch_size                         # Actualiza el total de muestras procesadas
        running_loss += loss.item() * batch_size    # Acumula la pérdida ponderada por tamaño de batch
        running_correct += preds.eq(targets).sum().item()
        # Suma el número de aciertos (predicción == etiqueta)

        batch_loss = loss.item()                    # Pérdida media del batch actual
        batch_acc = preds.eq(targets).float().mean().item() * 100.0
        # Accuracy del batch actual en porcentaje

        running_loss_avg = running_loss / total     # Pérdida media acumulada hasta este batch
        running_acc = running_correct / total * 100.0
        # Accuracy media acumulada hasta este batch

        lr = optimizer.param_groups[0]["lr"]        # Tasa de aprendizaje actual del primer grupo de parámetros


        global_step += 1                            # Incrementa el contador global de pasos (batches vistos)

        shared_state.update_algo_state(
            epoch=epoch,                            # Actualiza época actual
            global_step=global_step,                # Paso global
            batch_in_epoch=batch_idx,               # Índice de batch en la época
            batch_loss=batch_loss,                  # Pérdida de este batch
            batch_acc=batch_acc,                    # Accuracy de este batch
            running_loss=running_loss_avg,          # Pérdida acumulada
            running_acc=running_acc,                # Accuracy acumulada
            grad_norm=grad_norm,                    # Norma del gradiente
            lr=lr,                                  # Learning rate actual
        )
        # El hilo de monitor podrá leer este estado para correlacionarlo con medidas de hardware


        if (batch_idx + 1) % 10 == 0:
            # Imprime información de progreso cada 10 batches
            print(
                f"[Train] Epoch {epoch} "
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss {batch_loss:.4f} "
                f"Acc {batch_acc:.2f}% "
                f"Running Loss {running_loss_avg:.4f} "
                f"Running Acc {running_acc:.2f}% "
                f"GradNorm {grad_norm:.2f} "
                f"LR {lr:.5f}"
            )

    epoch_loss = running_loss / total              # Pérdida media de la época completa
    epoch_acc = running_correct / total * 100.0    # Accuracy media de la época completa
    return epoch_loss, epoch_acc, global_step      # Devuelve métricas de la época y el global_step actualizado



@torch.no_grad()
def validate(model, loader, criterion, epoch, device, use_amp, amp_dtype):
    """
    Ejecuta la fase de validación sin calcular gradientes.
    Devuelve pérdida y accuracy medias sobre el conjunto de validación.
    """
    model.eval()                                   # Pone el modelo en modo evaluación (desactiva dropout, BN en eval)
    shared_state.update_algo_state(phase="val", epoch=epoch)
    # Informa al estado compartido de que estamos en fase de validación


    running_loss = 0.0                             # Acumulador de pérdida total en validación
    running_correct = 0                            # Conteo de aciertos
    total = 0                                      # Conteo de muestras totales


    for batch_idx, (images, targets) in enumerate(loader):
        # Recorre los batches del DataLoader de validación
        images = images.to(device, non_blocking=True)    # Transfiere imágenes al dispositivo
        targets = targets.to(device, non_blocking=True)  # Transfiere etiquetas al dispositivo

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
            # Si AMP está activo, usamos autocast en validación también
        else:
            ctx = nullcontext()
            # Si no, contexto vacío

        with ctx:
            outputs = model(images)               # Forward pass en validación
            loss = criterion(outputs, targets)    # Pérdida de clasificación


        _, preds = outputs.max(1)                 # Predicción de clase por argmax
        batch_size = targets.size(0)              # Tamaño del batch
        total += batch_size                       # Actualiza total
        running_loss += loss.item() * batch_size  # Acumula pérdida ponderada
        running_correct += preds.eq(targets).sum().item()
        # Acumula número de aciertos


        batch_loss = loss.item()                  # Pérdida media del batch actual
        batch_acc = preds.eq(targets).float().mean().item() * 100.0
        # Accuracy del batch actual


        shared_state.update_algo_state(
            batch_in_epoch=batch_idx,             # Índice de batch de validación
            batch_loss=batch_loss,                # Pérdida de este batch
            batch_acc=batch_acc,                  # Accuracy de este batch
            running_loss=running_loss / total,    # Pérdida acumulada promedio
            running_acc=running_correct / total * 100.0,
            # Accuracy acumulada promedio
        )
        # Esto permite ver en el CSV cómo evoluciona validación en paralelo a las medidas de GPU

    epoch_loss = running_loss / total             # Pérdida media en todo el conjunto de validación
    epoch_acc = running_correct / total * 100.0   # Accuracy media en validación


    print(
        f"[Val]   Epoch {epoch} "
        f"Loss {epoch_loss:.4f} "
        f"Acc {epoch_acc:.2f}%"
    )                                             # Imprime resumen de validación
    return epoch_loss, epoch_acc                  # Devuelve métricas de validación para la época



@torch.no_grad()
def predict_all(model, loader, device, use_amp, amp_dtype):
    """
    Recorre un DataLoader y devuelve todos los labels verdaderos y predichos.
    Se usa al final para calcular la matriz de confusión y el F1.
    """
    model.eval()                                  # Modo evaluación
    all_preds = []                                # Lista de predicciones
    all_labels = []                               # Lista de etiquetas verdaderas

    for images, targets in loader:
        # Itera sobre todos los batches del DataLoader (típicamente el de validación)
        images = images.to(device, non_blocking=True)  # Pasa imágenes a dispositivo

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
            # Usa autocast si AMP está activado
        else:
            ctx = nullcontext()
            # Contexto vacío en caso contrario

        with ctx:
            outputs = model(images)               # Forward pass sobre el batch


        _, preds = outputs.max(1)                 # Extrae la clase ganadora por fila
        all_preds.extend(preds.cpu().numpy().tolist())
        # Convierte las predicciones a CPU, a NumPy y luego a lista y las añade a all_preds

        all_labels.extend(targets.numpy().tolist())
        # Convierte las etiquetas a lista y las añade a all_labels (targets ya están en CPU normalmente al salir del DataLoader)

    return np.array(all_labels), np.array(all_preds)
    # Devuelve dos arrays NumPy: y_true e y_pred para métricas globales


# =========================
# Parsing CSV de Nsight Compute
# =========================

def parse_ncu_csv(ncu_csv_path: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Lee un CSV generado por Nsight Compute (--csv --log-file ...) y construye
    un pequeño resumen con métricas SM agregadas.

    Suponemos métricas:
      - sm__warps_active.avg.pct_of_peak_sustained_active
      - sm__warps_active.avg
      - sm__inst_executed.sum
    Ajusta los nombres si usas otros en el comando ncu.
    """
    if ncu_csv_path is None:
        print("[SM PROFILE] ncu_csv_path es None, no se intenta leer perfil SM.")
        return None

    if not os.path.isfile(ncu_csv_path):
        print(f"[SM PROFILE] No se encontró CSV de Nsight Compute en: {ncu_csv_path}")
        return None

    occ_vals = []      # Lista para acumular valores de ocupación (por kernel o por fila del CSV)
    warps_vals = []    # Lista para acumular valores de warps activos
    inst_vals = []     # Lista para acumular conteos de instrucciones ejecutadas


    with open(ncu_csv_path, "r") as f:
        # Abre el archivo CSV generado por ncu
        reader = csv.DictReader(f)
        # Interpreta el CSV como diccionarios: cada fila es un dict con claves=nombre de columna
        for row in reader:
            # Recorre cada fila del CSV
            occ_key = "sm__warps_active.avg.pct_of_peak_sustained_active"
            # Nombre de columna esperado para porcentaje de ocupación de warps respecto al pico sostenido

            warps_key = "sm__warps_active.avg"
            # Nombre de columna esperado para warps activos promedio

            inst_key = "sm__inst_executed.sum"
            # Nombre de columna esperado para suma de instrucciones ejecutadas

            try:
                if occ_key in row and row[occ_key] not in ("", "N/A"):
                    occ_vals.append(float(row[occ_key]))
                    # Si la columna existe y no está vacía, la convierte a float y la acumula

                if warps_key in row and row[warps_key] not in ("", "N/A"):
                    warps_vals.append(float(row[warps_key]))
                    # Acumula warps activos promedio

                if inst_key in row and row[inst_key] not in ("", "N/A"):
                    inst_vals.append(float(row[inst_key]))
                    # Acumula instrucciones ejecutadas (por fila, normalmente por kernel o región)
            except ValueError:
                continue
                # Si hay un valor no convertible a float (texto raro), ignora esa fila y sigue

    if not occ_vals and not warps_vals and not inst_vals:
        print(f"[SM PROFILE] No se pudieron extraer métricas válidas de {ncu_csv_path}")
        return None

    def safe_mean(arr):
        return float(np.mean(arr)) if arr else None
         # Devuelve la media como float si el array no está vacío, o None si no hay datos

    def safe_p95(arr):
        return float(np.percentile(arr, 95)) if arr else None
   # Devuelve el percentil 95 si hay datos, o None si no hay datos

    sm_profile = {
        "ncu_csv_path": os.path.abspath(ncu_csv_path),
        # Guarda la ruta absoluta al CSV de Nsight Compute para trazabilidad

        "sm_occupancy_avg_pct": safe_mean(occ_vals),
        # Media de ocupación (porcentaje) a través de todas las filas del CSV

        "sm_occupancy_p95_pct": safe_p95(occ_vals),
        # Percentil 95 de ocupación; útil para capturar el "casi máximo" sin outliers extremos

        "warps_active_avg": safe_mean(warps_vals),
        # Media de warps activos a través de las filas

        "inst_executed_total": float(np.sum(inst_vals)) if inst_vals else None,
        # Suma total de instrucciones ejecutadas; depende del tipo exacto de métrica y del scope del CSV
    }


    print("[SM PROFILE] Resumen SM extraído de Nsight Compute:")
    # Imprime un encabezado informativo para la salida de consola
    for k, v in sm_profile.items():
        print(f"  {k}: {v}")
        # Imprime cada clave y valor del resumen para inspección rápida

    return sm_profile
    # Devuelve el diccionario con el resumen SM para incluirlo en experiment_summary.json



# =========================
# Main
# =========================

def main():
    config = parse_args() # Obtiene el diccionario de configuración desde la CLI
    set_seed(config["seed"]) # Fija semilla para reproducibilidad

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu") # Si hay GPU disponible, usa el device indicado (ej. cuda:0). Si no, usa CPU.
    print(f"Usando dispositivo: {device}")
    # Imprime el dispositivo final seleccionado

    print(f"Precisión: {config['precision']}")
    # Imprime la precisión solicitada (fp32/fp16/bf16)

    print(f"enable_sm_metrics: {config['enable_sm_metrics']}")
    # Imprime el flag de métricas SM embebidas (placeholder)

    print(f"ncu_csv_path: {config['ncu_csv_path']}")
    # Imprime la ruta al CSV de Nsight Compute si se proporcionó

    # Directorio del experimento
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construye un string de fecha/hora para un nombre único de experimento

    exp_dir = os.path.join(config["log_dir"], f"exp_{timestamp_str}")
    # Construye el directorio del experimento dentro de log_dir

    os.makedirs(exp_dir, exist_ok=True)
    # Crea el directorio si no existe

    print(f"Directorio de experimento: {exp_dir}")
    # Imprime la ruta donde se guardará todo


    # Guardar config en JSON
    config_path = os.path.join(exp_dir, "config.json") # Define la ruta del archivo config.json dentro del experimento
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)# Define la ruta del archivo config.json dentro del experimento
    print(f"Configuración guardada en: {config_path}") # Informa en consola dónde quedó el config.json

    # Precisión / AMP
    precision = config["precision"]
    # Extrae la precisión solicitada del config

    if device.type == "cuda":
        # Solo tiene sentido AMP cuando se usa GPU CUDA
        if precision == "fp16":
            amp_dtype = torch.float16
            # El dtype de autocast será float16
            use_amp = True
            # Activa autocast
            scaler = GradScaler(enabled=True)
            # Activa GradScaler para evitar underflow en FP16

        elif precision == "bf16":
            amp_dtype = torch.bfloat16
            # El dtype de autocast será bfloat16
            use_amp = True
            # Activa autocast
            scaler = GradScaler(enabled=False)
            # No se usa escalado de gradientes (bf16 suele ser estable sin scaler)

        else:
            amp_dtype = None
            # En fp32 no se usa autocast
            use_amp = False
            # Desactiva AMP
            scaler = GradScaler(enabled=False)
            # GradScaler queda desactivado

    else:
        amp_dtype = None
        # En CPU, se desactiva AMP en este diseño
        use_amp = False
        # No se usa autocast
        scaler = GradScaler(enabled=False)
        # No se usa scaler


    # Datos
    train_loader, val_loader, num_classes = get_dataloaders(
        config["data_dir"],
        config["batch_size"],
        config["num_workers"],
    )
    # Construye los DataLoaders y obtiene el número de clases


    # Modelo y optimizador
    model = get_model(num_classes).to(device)
    # Crea el modelo ConvNeXt adaptado al número de clases y lo mueve al dispositivo

    criterion = nn.CrossEntropyLoss()
    # Define la función de pérdida de clasificación multiclase

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    # Define el optimizador AdamW con lr y weight decay de la configuración

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"],
    )
    # Scheduler de coseno: ajusta lr a lo largo de las épocas hasta T_max


    # Monitorización
    gpu_monitor = GPUMonitorThread(
        device_index=0,
        interval=config["monitor_interval"],
        enable_sm_metrics=config["enable_sm_metrics"],
    )
    # Crea el hilo de monitorización para la GPU 0 con intervalo configurado

    gpu_monitor.start()
    # Arranca el hilo; desde aquí empieza a registrar muestras cada monitor_interval segundos

    epoch_train_loss = []
    # Lista para guardar pérdida media de entrenamiento por época

    epoch_train_acc = []
    # Lista para guardar accuracy media de entrenamiento por época

    epoch_val_loss = []
    # Lista para guardar pérdida media de validación por época

    epoch_val_acc = []
    # Lista para guardar accuracy media de validación por época

    epoch_metrics: List[Dict[str, Any]] = []
    # Lista de diccionarios con métricas por época (incluye tiempo y energía acumulada)

    global_step = 0
    # Contador global de pasos (batches)

    start_time = time.time()
    # Timestamp del inicio del entrenamiento para medir tiempo total

    try:
        # try/finally garantiza que el monitor se detenga incluso si hay excepción
        for epoch in range(1, config["num_epochs"] + 1):
            # Itera por épocas desde 1 hasta num_epochs inclusive
            t0 = time.time()
            # Tiempo inicial de la época actual

            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer,
                epoch, global_step, device,
                use_amp, amp_dtype, scaler,
            )# Ejecuta una época de entrenamiento y devuelve pérdida/accuracy medias y global_step actualizado

            val_loss, val_acc = validate(
                model, val_loader, criterion,
                epoch, device, use_amp, amp_dtype,
            ) # Ejecuta validación para la época actual y devuelve pérdida/accuracy

            scheduler.step()   # Actualiza el scheduler al final de la época

            epoch_train_loss.append(train_loss)
            # Guarda el train_loss de la época

            epoch_train_acc.append(train_acc)
            # Guarda el train_acc de la época

            epoch_val_loss.append(val_loss)
            # Guarda el val_loss de la época

            epoch_val_acc.append(val_acc)
            # Guarda el val_acc de la época


            elapsed = time.time() - t0
            # Calcula duración de la época en segundos

            energy_J = gpu_monitor.total_energy_joules()
            # Calcula energía acumulada total hasta el momento (julios)


            print(
                f"[Epoch {epoch}/{config['num_epochs']}] "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                f"Time {elapsed:.1f}s | "
                f"Energy {energy_J/3600:.3f} Wh (GPU, aprox)"
            )
            # Imprime resumen por época: métricas, tiempo y energía aproximada en Wh


            epoch_metrics.append({
                "epoch": epoch,
                # Número de época

                "train_loss": train_loss,
                # Pérdida media de entrenamiento

                "train_acc": train_acc,
                # Accuracy media de entrenamiento

                "val_loss": val_loss,
                # Pérdida media de validación

                "val_acc": val_acc,
                # Accuracy media de validación

                "epoch_time_s": elapsed,
                # Duración de la época en segundos

                "energy_Wh_cumulative": energy_J / 3600.0,
                # Energía acumulada hasta esta época (Wh)
            })
            # Añade el registro de métricas por época para el summary final


    finally:
        gpu_monitor.stop() # Detiene el monitor de GPU de forma segura (pase lo que pase)

    total_time = time.time() - start_time  # Tiempo total del entrenamiento en segundos
    total_energy_J = gpu_monitor.total_energy_joules() # Energía total aproximada en julios (integración sobre toda la ejecución)
    print(
        f"Entrenamiento terminado en {total_time/3600:.2f} h, "
        f"energía total GPU ~ {total_energy_J/3600:.2f} Wh"
    )  # Imprime resumen de tiempo total y energía total (Wh)

    # Guardar modelo
    model_path = os.path.join(exp_dir, "convnext_model.pth") # Ruta donde se guardará el checkpoint del modelo

    torch.save({
        "model_state_dict": model.state_dict(),  # Guarda los pesos del modelo
        "num_classes": num_classes,  # Guarda el número de clases para reconstrucción
        "config": config, # Guarda la configuración para trazabilidad
    }, model_path) # Serializa todo en un archivo .pth
    print(f"Modelo guardado en: {model_path}")   # Informa la ruta del modelo

    # CSV monitorización por segundo
    monitor_csv = os.path.join(exp_dir, "monitoring_per_second.csv") # Ruta del CSV de monitorización por segundo
    gpu_monitor.export_csv(monitor_csv) # Exporta todas las muestras del monitor a CSV
    print(f"CSV de monitorización guardado en: {monitor_csv}") # Informa la ruta del CSV de monitor

    # CSV por época
    epochs = list(range(1, config["num_epochs"] + 1)) # Lista de números de época para graficado y CSV

    epoch_csv = os.path.join(exp_dir, "training_epochs.csv")# Ruta del CSV de métricas por época
    with open(epoch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for e, tl, ta, vl, va in zip(
            epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc
        ):
            writer.writerow([e, tl, ta, vl, va])
    print(f"CSV de métricas por época guardado en: {epoch_csv}")

    # Métricas finales: matriz de confusión + F1
    y_true, y_pred = predict_all(model, val_loader, device, use_amp, amp_dtype)
    cm = confusion_matrix(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print("Matriz de confusión (val):")
    print(cm)
    print(f"F1 macro (val): {f1_macro:.4f}")

    # Gráficas de loss y accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_train_loss, label="Train Loss")
    plt.plot(epochs, epoch_val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pérdida por época")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_train_acc, label="Train Acc")
    plt.plot(epochs, epoch_val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy por época")
    plt.legend()

    loss_acc_fig = os.path.join(exp_dir, "loss_accuracy.png")
    plt.tight_layout()
    plt.savefig(loss_acc_fig, dpi=150)
    print(f"Gráfica de loss/accuracy guardada en: {loss_acc_fig}")
    plt.close()

    # Gráfica monitorización cruda
    monitor_fig = None
    if gpu_monitor.records:
        times = [r.rel_time_s for r in gpu_monitor.records]
        power = [r.power_W for r in gpu_monitor.records]
        util_gpu = [r.util_gpu for r in gpu_monitor.records]
        mem = [r.mem_used_MB for r in gpu_monitor.records]
        temp = [r.temp_C for r in gpu_monitor.records]

        plt.figure(figsize=(10, 6))
        plt.plot(times, power, label="Power (W)")
        plt.plot(times, util_gpu, label="Util GPU (%)")
        plt.plot(times, mem, label="Mem Used (MB)")
        plt.plot(times, temp, label="Temp (°C)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Valor")
        plt.title("Monitorización por segundo (valores crudos)")
        plt.legend()
        monitor_fig = os.path.join(exp_dir, "monitoring_raw.png")
        plt.tight_layout()
        plt.savefig(monitor_fig, dpi=150)
        print(f"Gráfica de monitorización guardada en: {monitor_fig}")
        plt.close()

    # Perfil SM a partir de Nsight Compute (si hay CSV)
    sm_profile = parse_ncu_csv(config.get("ncu_csv_path"))

    # =========================
    # experiment_summary.json
    # =========================

    exp_summary = {
        "config": config,
        "num_classes": num_classes,
        "total_time_s": total_time,
        "total_energy_J": total_energy_J,
        "total_energy_Wh": total_energy_J / 3600.0,
        "final_f1_macro": float(f1_macro),
        "final_confusion_matrix": cm.tolist(),
        "epoch_metrics": epoch_metrics,
        "paths": {
            "model": model_path,
            "monitor_csv": monitor_csv,
            "epoch_csv": epoch_csv,
            "loss_acc_fig": loss_acc_fig,
            "monitor_fig": monitor_fig,
            "config_json": config_path,
            "ncu_csv": config.get("ncu_csv_path"),
        },
        "sm_profile": sm_profile,
        "timestamp_start": datetime.datetime.fromtimestamp(shared_state.start_time).isoformat(),
        "timestamp_end": datetime.datetime.now().isoformat(),
    }

    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=4)
    print(f"Resumen de experimento guardado en: {summary_path}") # Informa dónde quedó el summary


if __name__ == "__main__":
    main()  # Si este archivo se ejecuta como script principal, llama a main()
