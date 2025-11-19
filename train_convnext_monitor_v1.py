import os              # Módulo para operaciones con el sistema de archivos
import time            # Módulo para medir tiempos y timestamps
import threading       # Para crear hilos (threads) de monitorización
import math            # Funciones matemáticas (log10, sqrt, etc.)
import csv             # Para escribir archivos CSV
import json            # Para guardar configuración y resultados en JSON
import argparse        # Para parsear argumentos de línea de comandos
import random          # Para semilla aleatoria de Python
import datetime        # Para manejar fechas y timestamps legibles
from dataclasses import dataclass, asdict  # Para definir estructuras de datos simples
from typing import Optional, List, Dict, Any  # Tipado opcional para claridad

import numpy as np                     # Numpy para operaciones numéricas
import torch                           # PyTorch, framework principal
import torch.nn as nn                  # Módulo para redes neuronales
import torch.optim as optim            # Optimizadores (AdamW, etc.)
from torch.utils.data import DataLoader  # Para cargar datos por batches
from torchvision import datasets, transforms, models  # Datasets, transformaciones y modelos predefinidos

import pynvml                         # Interfaz Python para NVML (NVIDIA), monitorización GPU
import matplotlib.pyplot as plt       # Para graficar
from sklearn.metrics import confusion_matrix, f1_score  # Métricas de evaluación (matriz de confusión, F1)


# =========================
# Configuración por defecto
# =========================

CONFIG_DEFAULT = {
    "data_dir": "/ruta/a/tu/imagenet_10gb",  # Ruta por defecto al dataset (ImageNet reducido)
    "log_dir": "./logs_convnext",            # Directorio base para guardar resultados
    "batch_size": 128,                       # Tamaño de batch para entrenamiento
    "num_workers": 8,                        # Trabajadores del DataLoader (hilos de lectura de datos)
    "num_epochs": 20,                        # Número de épocas de entrenamiento
    "lr": 1e-3,                              # Tasa de aprendizaje inicial
    "weight_decay": 0.05,                    # Decaimiento L2 de pesos (regularización)
    "monitor_interval": 1.0,                 # Intervalo (segundos) para tomar muestras de monitorización
    "seed": 1234,                            # Semilla para reproducibilidad
    "device": "cuda:0",                      # Dispositivo por defecto (GPU 0 si está disponible)
}


# =========================
# Parsing de argumentos
# =========================

def parse_args() -> Dict[str, Any]:
    # Crea un parser para leer argumentos desde la línea de comandos
    parser = argparse.ArgumentParser(
        description="Entrenamiento ConvNeXt con monitorización y tracking JSON"
    )
    # Argumento: ruta al dataset
    parser.add_argument("--data_dir", type=str, default=CONFIG_DEFAULT["data_dir"])
    # Argumento: ruta base para logs
    parser.add_argument("--log_dir", type=str, default=CONFIG_DEFAULT["log_dir"])
    # Argumento: tamaño de batch
    parser.add_argument("--batch_size", type=int, default=CONFIG_DEFAULT["batch_size"])
    # Argumento: número de workers del DataLoader
    parser.add_argument("--num_workers", type=int, default=CONFIG_DEFAULT["num_workers"])
    # Argumento: número de épocas
    parser.add_argument("--epochs", type=int, default=CONFIG_DEFAULT["num_epochs"])
    # Argumento: learning rate
    parser.add_argument("--lr", type=float, default=CONFIG_DEFAULT["lr"])
    # Argumento: weight decay
    parser.add_argument("--weight_decay", type=float, default=CONFIG_DEFAULT["weight_decay"])
    # Argumento: intervalo de monitorización en segundos
    parser.add_argument("--monitor_interval", type=float, default=CONFIG_DEFAULT["monitor_interval"])
    # Argumento: semilla para reproducibilidad
    parser.add_argument("--seed", type=int, default=CONFIG_DEFAULT["seed"])
    # Argumento: dispositivo (cuda:0, cuda, cpu, etc.)
    parser.add_argument("--device", type=str, default=CONFIG_DEFAULT["device"])
    # Parsea los argumentos pasados por CLI
    args = parser.parse_args()

    # Construye un diccionario de configuración a partir de los argumentos
    config = {
        "data_dir": args.data_dir,
        "log_dir": args.log_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "monitor_interval": args.monitor_interval,
        "seed": args.seed,
        "device": args.device,
    }
    return config  # Devuelve la configuración


# =========================
# Semilla global
# =========================

def set_seed(seed: int):
    # Fija la semilla para el módulo random de Python
    random.seed(seed)
    # Fija la semilla para NumPy
    np.random.seed(seed)
    # Fija la semilla para PyTorch en CPU
    torch.manual_seed(seed)
    # Fija la semilla para todas las GPUs (si hay)
    torch.cuda.manual_seed_all(seed)
    # Fuerza comportamiento determinista en cudnn (más reproducible pero algo más lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Estado compartido
# =========================

@dataclass
class AlgoState:
    # Representa el estado del algoritmo (entrenamiento) en un instante dado
    timestamp: float = 0.0        # Timestamp de la última actualización
    epoch: int = 0                # Época actual
    global_step: int = 0          # Paso global (número de batch acumulado)
    batch_in_epoch: int = 0       # Índice de batch dentro de la época
    phase: str = "idle"           # Fase: "train", "val" o "idle"
    batch_loss: float = math.nan  # Pérdida del último batch
    batch_acc: float = math.nan   # Accuracy del último batch
    running_loss: float = math.nan  # Pérdida promedio acumulada en la época
    running_acc: float = math.nan   # Accuracy promedio acumulado en la época
    grad_norm: float = math.nan     # Norma global del gradiente
    lr: float = math.nan            # Learning rate actual


@dataclass
class MonitorRecord:
    # Registro de monitorización combinando hardware + estado de algoritmo
    timestamp: float          # Timestamp absoluto
    rel_time_s: float         # Tiempo relativo desde el inicio del experimento
    power_W: float            # Potencia instantánea (W)
    util_gpu: float           # Utilización de GPU (%)
    util_mem: float           # Utilización de memoria (%)
    mem_used_MB: float        # Memoria usada (MB)
    temp_C: float             # Temperatura de la GPU (ºC)

    # Estado del algoritmo asociado a este instante
    epoch: int
    global_step: int
    batch_in_epoch: int
    phase: str
    batch_loss: float
    batch_acc: float
    running_loss: float
    running_acc: float
    grad_norm: float
    lr: float

    # Campos para métricas avanzadas SM (para el futuro con CUPTI / Nsight)
    sm_occupancy: Optional[float] = None
    active_warps: Optional[float] = None
    inst_executed: Optional[float] = None
    stalls: Optional[float] = None


class SharedState:
    # Clase que mantiene el estado compartido entre el hilo de entrenamiento y el de monitorización
    def __init__(self):
        self._lock = threading.Lock()           # Lock para evitar condiciones de carrera
        self._algo_state = AlgoState()          # Estado actual del algoritmo
        self._start_time = time.time()          # Timestamp de inicio del experimento

    def update_algo_state(self, **kwargs):
        # Actualiza el estado del algoritmo con nuevos valores
        with self._lock:                        # Asegura exclusión mutua
            for k, v in kwargs.items():         # Recorre los campos a actualizar
                if hasattr(self._algo_state, k):
                    setattr(self._algo_state, k, v)  # Actualiza el campo
            self._algo_state.timestamp = time.time()  # Actualiza el timestamp

    def snapshot(self) -> (AlgoState, float):
        # Devuelve una copia del estado actual y el tiempo relativo desde el inicio
        with self._lock:
            algo = AlgoState(**asdict(self._algo_state))  # Copia del dataclass
        rel_t = algo.timestamp - self._start_time         # Tiempo relativo
        return algo, rel_t                                # Devuelve ambas cosas

    @property
    def start_time(self):
        # Devuelve el tiempo de inicio del experimento
        return self._start_time


# Instancia global del estado compartido
shared_state = SharedState()


# =========================
# Monitor GPU + estado
# =========================

class GPUMonitorThread(threading.Thread):
    # Hilo que monitoriza GPU + estado del algoritmo cada cierto intervalo
    def __init__(self, device_index=0, interval=1.0):
        super().__init__()                  # Inicializa la clase base Thread
        self.device_index = device_index    # Índice de GPU a monitorizar
        self.interval = interval            # Intervalo entre muestras (segundos)
        self._stop_flag = threading.Event() # Flag para indicar parada
        self.records: List[MonitorRecord] = []  # Lista de registros de monitorización

        pynvml.nvmlInit()                               # Inicializa NVML
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)  # Handle de la GPU

    def run(self):
        # Método principal del hilo: bucle de monitorización
        while not self._stop_flag.is_set():   # Mientras no se haya pedido parar
            now = time.time()                 # Timestamp actual
            algo_state, rel_t = shared_state.snapshot()  # Copia del estado del algoritmo y tiempo relativo

            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)   # Potencia en mW
            power_W = power_mW / 1000.0                              # Pasa a W
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle) # Utilización GPU/mem
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )                                                         # Temperatura GPU
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)    # Info de memoria
            mem_used_MB = mem_info.used / (1024 ** 2)                 # Memoria usada en MB

            # Hooks para integrar CUPTI / Nsight y rellenar estas métricas más adelante
            sm_occupancy = None
            active_warps = None
            inst_executed = None
            stalls = None

            # Crea un registro de monitorización uniendo HW + estado algoritmo
            rec = MonitorRecord(
                timestamp=now,
                rel_time_s=rel_t,
                power_W=power_W,
                util_gpu=util.gpu,
                util_mem=util.memory,
                mem_used_MB=mem_used_MB,
                temp_C=temp,
                epoch=algo_state.epoch,
                global_step=algo_state.global_step,
                batch_in_epoch=algo_state.batch_in_epoch,
                phase=algo_state.phase,
                batch_loss=algo_state.batch_loss,
                batch_acc=algo_state.batch_acc,
                running_loss=algo_state.running_loss,
                running_acc=algo_state.running_acc,
                grad_norm=algo_state.grad_norm,
                lr=algo_state.lr,
                sm_occupancy=sm_occupancy,
                active_warps=active_warps,
                inst_executed=inst_executed,
                stalls=stalls,
            )
            self.records.append(rec)   # Añade el registro a la lista

            time.sleep(self.interval)  # Espera hasta la siguiente muestra

    def stop(self):
        # Señala que el hilo debe parar y espera a que termine
        self._stop_flag.set()
        self.join()
        pynvml.nvmlShutdown()  # Cierra NVML

    def total_energy_joules(self) -> float:
        # Calcula la energía aproximada integrada (trapecios) a partir de la potencia
        if len(self.records) < 2:
            return 0.0
        E = 0.0
        for i in range(len(self.records) - 1):
            p1 = self.records[i].power_W          # Potencia en t1
            p2 = self.records[i + 1].power_W      # Potencia en t2
            t1 = self.records[i].timestamp        # Tiempo t1
            t2 = self.records[i + 1].timestamp    # Tiempo t2
            E += 0.5 * (p1 + p2) * (t2 - t1)      # Área del trapecio (J = W * s)
        return E

    def export_csv(self, path: str):
        # Exporta todos los registros de monitorización a un CSV
        if not self.records:
            return
        fieldnames = list(asdict(self.records[0]).keys())  # Nombres de columnas del dataclass
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)  # Crea escritor CSV
            writer.writeheader()                               # Escribe cabecera
            for rec in self.records:
                writer.writerow(asdict(rec))                   # Escribe cada registro como dict


# =========================
# Datos y modelo
# =========================

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    # Normalización estándar para modelos pre-entrenados en ImageNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Transformaciones para entrenamiento (augmentación)
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),   # Recorte aleatorio y resize a 224x224
        transforms.RandomHorizontalFlip(),   # Flip horizontal aleatorio
        transforms.ToTensor(),               # Convertir a tensor
        normalize,                           # Normalizar
    ])
    # Transformaciones para validación (determinísticas)
    val_tfms = transforms.Compose([
        transforms.Resize(256),              # Redimensionar lado corto a 256
        transforms.CenterCrop(224),          # Recorte centrado 224x224
        transforms.ToTensor(),               # A tensor
        normalize,                           # Normalizar
    ])

    # Carga dataset de entrenamiento asumiendo estructura ImageFolder
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    # Carga dataset de validación
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    # DataLoader de entrenamiento
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    # DataLoader de validación
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    num_classes = len(train_ds.classes)  # Número de clases (carpetas)
    return train_loader, val_loader, num_classes


def get_model(num_classes: int):
    # Carga ConvNeXt-Tiny preentrenado en ImageNet
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    # Obtiene número de features de la última capa
    in_features = model.classifier[2].in_features
    # Reemplaza la última capa por una adaptada a num_classes
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# =========================
# Entrenamiento
# =========================

def compute_grad_norm(model: nn.Module) -> float:
    # Calcula la norma L2 global de todos los gradientes del modelo
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)        # Norma L2 de ese tensor
            total_norm_sq += param_norm.item() ** 2 # Acumula el cuadrado
    return math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0


def train_one_epoch(
    model, loader, criterion, optimizer, epoch, global_step, device
):
    # Pone el modelo en modo entrenamiento
    model.train()
    # Actualiza el estado compartido indicando que estamos en fase "train"
    shared_state.update_algo_state(phase="train", epoch=epoch)

    running_loss = 0.0       # Pérdida acumulada
    running_correct = 0      # Aciertos acumulados
    total = 0                # Número total de muestras vistas

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)   # Mueve imágenes a GPU/CPU
        targets = targets.to(device, non_blocking=True) # Mueve etiquetas a GPU/CPU

        optimizer.zero_grad()                  # Pone gradientes a cero
        outputs = model(images)                # Forward pass
        loss = criterion(outputs, targets)     # Calcula pérdida
        loss.backward()                        # Backprop

        grad_norm = compute_grad_norm(model)   # Calcula norma del gradiente
        optimizer.step()                       # Actualiza parámetros

        _, preds = outputs.max(1)              # Obtiene predicciones (clase con máximo)
        batch_size = targets.size(0)           # Tamaño del batch
        total += batch_size                    # Actualiza contador de muestras
        running_loss += loss.item() * batch_size              # Acumula pérdida total
        running_correct += preds.eq(targets).sum().item()     # Acumula aciertos

        batch_loss = loss.item()                              # Pérdida del batch
        batch_acc = preds.eq(targets).float().mean().item() * 100.0  # Acc del batch (%)
        running_loss_avg = running_loss / total               # Pérdida promedio acumulada
        running_acc = running_correct / total * 100.0         # Accuracy promedio acumulada
        lr = optimizer.param_groups[0]["lr"]                  # LR actual

        global_step += 1                                      # Incrementa step global

        # Actualiza el estado compartido con los valores del último batch
        shared_state.update_algo_state(
            epoch=epoch,
            global_step=global_step,
            batch_in_epoch=batch_idx,
            batch_loss=batch_loss,
            batch_acc=batch_acc,
            running_loss=running_loss_avg,
            running_acc=running_acc,
            grad_norm=grad_norm,
            lr=lr,
        )

        # Imprime en consola cada 10 batches
        if (batch_idx + 1) % 10 == 0:
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

    epoch_loss = running_loss / total          # Pérdida promedio de la época
    epoch_acc = running_correct / total * 100.0  # Accuracy promedio de la época
    return epoch_loss, epoch_acc, global_step   # Devuelve métricas y último step global


@torch.no_grad()
def validate(model, loader, criterion, epoch, device):
    # Función de validación (sin gradientes)
    model.eval()                                    # Modo evaluación
    shared_state.update_algo_state(phase="val", epoch=epoch)  # Indica fase "val"

    running_loss = 0.0             # Pérdida acumulada
    running_correct = 0            # Aciertos acumulados
    total = 0                      # Total de muestras

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)   # A dispositivo
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)                         # Forward
        loss = criterion(outputs, targets)              # Pérdida

        _, preds = outputs.max(1)                       # Predicciones
        batch_size = targets.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        running_correct += preds.eq(targets).sum().item()

        batch_loss = loss.item()
        batch_acc = preds.eq(targets).float().mean().item() * 100.0

        # Actualiza estado compartido con info de este batch de validación
        shared_state.update_algo_state(
            batch_in_epoch=batch_idx,
            batch_loss=batch_loss,
            batch_acc=batch_acc,
            running_loss=running_loss / total,
            running_acc=running_correct / total * 100.0,
        )

    epoch_loss = running_loss / total        # Pérdida promedio en validación
    epoch_acc = running_correct / total * 100.0  # Accuracy promedio en validación

    # Imprime resumen de validación
    print(
        f"[Val]   Epoch {epoch} "
        f"Loss {epoch_loss:.4f} "
        f"Acc {epoch_acc:.2f}%"
    )
    return epoch_loss, epoch_acc             # Devuelve métricas de validación


@torch.no_grad()
def predict_all(model, loader, device):
    # Genera predicciones para todo un DataLoader (para matriz de confusión, F1, etc.)
    model.eval()
    all_preds = []    # Lista de predicciones
    all_labels = []   # Lista de etiquetas verdaderas
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(targets.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)  # Devuelve arrays NumPy


# =========================
# Main
# =========================

def main():
    config = parse_args()                 # Lee configuración desde CLI + defaults
    set_seed(config["seed"])              # Fija la semilla global

    # Selecciona dispositivo (prefiere GPU si está disponible)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Crea un directorio único para el experimento con timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config["log_dir"], f"exp_{timestamp_str}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Directorio de experimento: {exp_dir}")

    # Guarda la configuración inicial en JSON
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuración guardada en: {config_path}")

    # Carga los dataloaders
    train_loader, val_loader, num_classes = get_dataloaders(
        config["data_dir"],
        config["batch_size"],
        config["num_workers"],
    )

    # Construye el modelo, la pérdida y el optimizador
    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    # Scheduler de LR (Cosine Annealing a lo largo de las épocas)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"],
    )

    # Crea y arranca el hilo de monitorización de GPU + estado
    gpu_monitor = GPUMonitorThread(
        device_index=0,
        interval=config["monitor_interval"],
    )
    gpu_monitor.start()

    # Listas para guardar métricas por época
    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []

    # Lista para un tracking más detallado por época
    epoch_metrics: List[Dict[str, Any]] = []

    global_step = 0                # Contador global de batches
    start_time = time.time()       # Tiempo de inicio del experimento

    try:
        # Bucle principal de entrenamiento por épocas
        for epoch in range(1, config["num_epochs"] + 1):
            t0 = time.time()  # Tiempo de inicio de la época

            # Entrena una época completa
            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch, global_step, device
            )
            # Evalúa en validación
            val_loss, val_acc = validate(
                model, val_loader, criterion, epoch, device
            )

            # Actualiza scheduler de LR
            scheduler.step()

            # Guarda métricas de la época en listas
            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)

            elapsed = time.time() - t0          # Tiempo que ha tardado la época
            energy_J = gpu_monitor.total_energy_joules()  # Energía total acumulada hasta ahora

            # Imprime resumen de la época
            print(
                f"[Epoch {epoch}/{config['num_epochs']}] "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                f"Time {elapsed:.1f}s | "
                f"Energy {energy_J/3600:.3f} Wh (GPU, aprox)"
            )

            # Guarda las métricas de esta época en una estructura para el JSON
            epoch_metrics.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": elapsed,
                "energy_Wh_cumulative": energy_J / 3600.0,
            })

    finally:
        # Asegura que paramos la monitorización incluso si hay excepción
        gpu_monitor.stop()

    total_time = time.time() - start_time          # Tiempo total del experimento
    total_energy_J = gpu_monitor.total_energy_joules()  # Energía total estimada
    print(
        f"Entrenamiento terminado en {total_time/3600:.2f} h, "
        f"energía total GPU ~ {total_energy_J/3600:.2f} Wh"
    )

    # Guarda el modelo entrenado en un archivo .pth
    model_path = os.path.join(exp_dir, "convnext_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "config": config,
    }, model_path)
    print(f"Modelo guardado en: {model_path}")

    # Exporta el CSV de monitorización por segundo
    monitor_csv = os.path.join(exp_dir, "monitoring_per_second.csv")
    gpu_monitor.export_csv(monitor_csv)
    print(f"CSV de monitorización guardado en: {monitor_csv}")

    # Exporta un CSV de métricas por época
    epochs = list(range(1, config["num_epochs"] + 1))
    epoch_csv = os.path.join(exp_dir, "training_epochs.csv")
    with open(epoch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for e, tl, ta, vl, va in zip(
            epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc
        ):
            writer.writerow([e, tl, ta, vl, va])
    print(f"CSV de métricas por época guardado en: {epoch_csv}")

    # Calcula matriz de confusión y F1 macro sobre el conjunto de validación
    y_true, y_pred = predict_all(model, val_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print("Matriz de confusión (val):")
    print(cm)
    print(f"F1 macro (val): {f1_macro:.4f}")

    # Dibuja y guarda las gráficas de pérdida y accuracy por época
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

    # Dibuja y guarda la gráfica de monitorización por segundo en escala log
    monitor_fig = None
    if gpu_monitor.records:
        times = [r.rel_time_s for r in gpu_monitor.records]  # Tiempos relativos
        power = [r.power_W for r in gpu_monitor.records]     # Potencia
        util_gpu = [r.util_gpu for r in gpu_monitor.records] # Utilización GPU
        mem = [r.mem_used_MB for r in gpu_monitor.records]   # Memoria usada
        temp = [r.temp_C for r in gpu_monitor.records]       # Temperatura

        eps = 1e-3  # Pequeño epsilon para evitar log10(0)
        power_log = [math.log10(p + eps) for p in power]
        util_log = [math.log10(u + eps) for u in util_gpu]
        mem_log = [math.log10(m + eps) for m in mem]
        temp_log = [math.log10(t + eps) for t in temp]

        plt.figure(figsize=(10, 6))
        plt.plot(times, power_log, label="log10(Power W)")
        plt.plot(times, util_log, label="log10(Util GPU %)")
        plt.plot(times, mem_log, label="log10(Mem Used MB)")
        plt.plot(times, temp_log, label="log10(Temp C)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("log10(magnitud)")
        plt.title("Monitorización por segundo (normalizada, escala log10)")
        plt.legend()
        monitor_fig = os.path.join(exp_dir, "monitoring_log.png")
        plt.tight_layout()
        plt.savefig(monitor_fig, dpi=150)
        print(f"Gráfica de monitorización guardada en: {monitor_fig}")
        plt.close()

    # =========================
    # Tracking de experimento en JSON
    # =========================

    # Construye un diccionario resumen del experimento
    exp_summary = {
        "config": config,
        "num_classes": num_classes,
        "total_time_s": total_time,
        "total_energy_J": total_energy_J,
        "total_energy_Wh": total_energy_J / 3600.0,
        "final_f1_macro": float(f1_macro),
        "final_confusion_matrix": cm.tolist(),  # matriz de confusión como lista
        "epoch_metrics": epoch_metrics,         # métricas por época
        "paths": {
            "model": model_path,
            "monitor_csv": monitor_csv,
            "epoch_csv": epoch_csv,
            "loss_acc_fig": loss_acc_fig,
            "monitor_fig": monitor_fig,
            "config_json": config_path,
        },
        "timestamp_start": datetime.datetime.fromtimestamp(shared_state.start_time).isoformat(),
        "timestamp_end": datetime.datetime.now().isoformat(),
    }

    # Guarda el resumen del experimento en JSON
    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=4)
    print(f"Resumen de experimento guardado en: {summary_path}")


if __name__ == "__main__":
    # Punto de entrada del script
    main()
