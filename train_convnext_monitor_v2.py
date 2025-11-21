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
from contextlib import nullcontext

import numpy as np                     # Numpy para operaciones numéricas
import torch                           # PyTorch, framework principal
import torch.nn as nn                  # Módulo para redes neuronales
import torch.optim as optim            # Optimizadores (AdamW, etc.)
from torch.utils.data import DataLoader  # Para cargar datos por batches
from torchvision import datasets, transforms, models  # Datasets, transformaciones y modelos predefinidos

import pynvml                         # Interfaz Python para NVML (NVIDIA), monitorización GPU
import matplotlib.pyplot as plt       # Para graficar
from sklearn.metrics import confusion_matrix, f1_score  # Métricas de evaluación (matriz de confusión, F1)
from torch.cuda.amp import autocast, GradScaler


# =========================
# Configuración por defecto
# =========================

CONFIG_DEFAULT = {
    "data_dir": "./imagenet1k",     # carpeta que contiene train/ y val/
    "log_dir": "./logs_convnext",            # Directorio base para guardar resultados
    "batch_size": 128,                       # Tamaño de batch para entrenamiento
    "num_workers": 8,                        # Trabajadores del DataLoader (hilos de lectura de datos)
    "num_epochs": 20,                        # Número de épocas de entrenamiento
    "lr": 1e-3,                              # Tasa de aprendizaje inicial
    "weight_decay": 0.05,                    # Decaimiento L2 de pesos (regularización)
    "monitor_interval": 1.0,                 # Intervalo (segundos) para tomar muestras de monitorización
    "seed": 1234,                            # Semilla para reproducibilidad
    "device": "cuda:0",                      # Dispositivo por defecto (GPU 0 si está disponible)
    "precision": "fp32",            # "fp32", "fp16", "bf16"
    "enable_sm_metrics": False,     # activar métricas SM avanzadas (placeholder)
}


# =========================
# Parsing de argumentos CLI
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
    parser.add_argument(
        "--precision",
        type=str,
        default=CONFIG_DEFAULT["precision"],
        choices=["fp32", "fp16", "bf16"],
        help="Precisión numérica: fp32, fp16 o bf16",
    )
    parser.add_argument(
        "--enable_sm_metrics",
        action="store_true",
        help="Si se especifica, intenta recoger métricas SM (occupancy, warps, etc.).",
    )
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
        "precision": args.precision,
        "enable_sm_metrics": args.enable_sm_metrics,
    }
    return config # Devuelve la configuración


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
    # Fija la semilla para todas las GPUs 
    torch.cuda.manual_seed_all(seed)
     # Fuerza comportamiento determinista en cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Estado compartido
# =========================

@dataclass
class AlgoState:
    # Representa el estado del algoritmo (entrenamiento) en un instante dado
    timestamp: float = 0.0  # Timestamp de la última actualización
    epoch: int = 0 # Época actual
    global_step: int = 0 # Paso global (número de batch acumulado)
    batch_in_epoch: int = 0 # Índice de batch dentro de la época
    phase: str = "idle"      # "train" / "val" / "idle"
    batch_loss: float = math.nan # Pérdida del último batch
    batch_acc: float = math.nan # Accuracy del último batch
    running_loss: float = math.nan # Pérdida promedio acumulada en la época
    running_acc: float = math.nan # Accuracy promedio acumulado en la época
    grad_norm: float = math.nan # Norma global del gradiente
    lr: float = math.nan # Learning rate actual


@dataclass
class MonitorRecord:
    timestamp: float
    rel_time_s: float
    power_W: float
    util_gpu: float
    util_mem: float
    mem_used_MB: float
    temp_C: float

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

    # Métricas SM avanzadas (se rellenan solo si enable_sm_metrics=True y se implementa _collect_sm_metrics)
    sm_occupancy: Optional[float] = None
    active_warps: Optional[float] = None
    inst_executed: Optional[float] = None
    stalls: Optional[float] = None


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._algo_state = AlgoState()
        self._start_time = time.time()

    def update_algo_state(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._algo_state, k):
                    setattr(self._algo_state, k, v)
            self._algo_state.timestamp = time.time()

    def snapshot(self) -> (AlgoState, float):
        with self._lock:
            algo = AlgoState(**asdict(self._algo_state))
        rel_t = algo.timestamp - self._start_time
        return algo, rel_t

    @property
    def start_time(self):
        return self._start_time


shared_state = SharedState()


# =========================
# Monitor GPU + estado
# =========================

class GPUMonitorThread(threading.Thread):
    def __init__(self, device_index=0, interval=1.0, enable_sm_metrics=False):
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.enable_sm_metrics = enable_sm_metrics
        self._stop_flag = threading.Event()
        self.records: List[MonitorRecord] = []

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def _collect_sm_metrics(self):
        """
        Punto de extensión para integrar CUPTI / Nsight.
        De momento devuelve None en todo para no añadir overhead extra.
        """
        return None, None, None, None

    def run(self):
        while not self._stop_flag.is_set():
            now = time.time()
            algo_state, rel_t = shared_state.snapshot()

            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_W = power_mW / 1000.0
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used_MB = mem_info.used / (1024 ** 2)

            if self.enable_sm_metrics:
                sm_occupancy, active_warps, inst_executed, stalls = self._collect_sm_metrics()
            else:
                sm_occupancy = None
                active_warps = None
                inst_executed = None
                stalls = None

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
            self.records.append(rec)

            time.sleep(self.interval)

    def stop(self):
        self._stop_flag.set()
        self.join()
        pynvml.nvmlShutdown()

    def total_energy_joules(self) -> float:
        if len(self.records) < 2:
            return 0.0
        E = 0.0
        for i in range(len(self.records) - 1):
            p1 = self.records[i].power_W
            p2 = self.records[i + 1].power_W
            t1 = self.records[i].timestamp
            t2 = self.records[i + 1].timestamp
            E += 0.5 * (p1 + p2) * (t2 - t1)
        return E

    def export_csv(self, path: str):
        if not self.records:
            return
        fieldnames = list(asdict(self.records[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))


# =========================
# Datos y modelo
# =========================

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    """
    data_dir debe contener:
      - train/
      - val/
    con subcarpetas por clase.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    num_classes = len(train_ds.classes)
    return train_loader, val_loader, num_classes


def get_model(num_classes: int):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# =========================
# Entrenamiento
# =========================

def compute_grad_norm(model: nn.Module) -> float:
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0


def train_one_epoch(
    model, loader, criterion, optimizer,
    epoch, global_step, device,
    use_amp, amp_dtype, scaler,
):
    model.train()
    shared_state.update_algo_state(phase="train", epoch=epoch)

    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
        else:
            ctx = nullcontext()

        with ctx:
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = compute_grad_norm(model)
            optimizer.step()

        _, preds = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        running_correct += preds.eq(targets).sum().item()

        batch_loss = loss.item()
        batch_acc = preds.eq(targets).float().mean().item() * 100.0
        running_loss_avg = running_loss / total
        running_acc = running_correct / total * 100.0
        lr = optimizer.param_groups[0]["lr"]

        global_step += 1

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

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total * 100.0
    return epoch_loss, epoch_acc, global_step


@torch.no_grad()
def validate(model, loader, criterion, epoch, device, use_amp, amp_dtype):
    model.eval()
    shared_state.update_algo_state(phase="val", epoch=epoch)

    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
        else:
            ctx = nullcontext()

        with ctx:
            outputs = model(images)
            loss = criterion(outputs, targets)

        _, preds = outputs.max(1)
        batch_size = targets.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        running_correct += preds.eq(targets).sum().item()

        batch_loss = loss.item()
        batch_acc = preds.eq(targets).float().mean().item() * 100.0

        shared_state.update_algo_state(
            batch_in_epoch=batch_idx,
            batch_loss=batch_loss,
            batch_acc=batch_acc,
            running_loss=running_loss / total,
            running_acc=running_correct / total * 100.0,
        )

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total * 100.0

    print(
        f"[Val]   Epoch {epoch} "
        f"Loss {epoch_loss:.4f} "
        f"Acc {epoch_acc:.2f}%"
    )
    return epoch_loss, epoch_acc


@torch.no_grad()
def predict_all(model, loader, device, use_amp, amp_dtype):
    model.eval()
    all_preds = []
    all_labels = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
        else:
            ctx = nullcontext()

        with ctx:
            outputs = model(images)

        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(targets.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


# =========================
# Main
# =========================

def main():
    config = parse_args()
    set_seed(config["seed"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    print(f"Precisión: {config['precision']}")
    print(f"enable_sm_metrics: {config['enable_sm_metrics']}")

    # Directorio del experimento
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config["log_dir"], f"exp_{timestamp_str}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Directorio de experimento: {exp_dir}")

    # Guardar config en JSON
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuración guardada en: {config_path}")

    # Precisión / AMP
    precision = config["precision"]
    if device.type == "cuda":
        if precision == "fp16":
            amp_dtype = torch.float16
            use_amp = True
            scaler = GradScaler(enabled=True)
        elif precision == "bf16":
            amp_dtype = torch.bfloat16
            use_amp = True
            scaler = GradScaler(enabled=False)
        else:
            amp_dtype = None
            use_amp = False
            scaler = GradScaler(enabled=False)
    else:
        amp_dtype = None
        use_amp = False
        scaler = GradScaler(enabled=False)

    # Datos
    train_loader, val_loader, num_classes = get_dataloaders(
        config["data_dir"],
        config["batch_size"],
        config["num_workers"],
    )

    # Modelo y optimizador
    model = get_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"],
    )

    # Monitorización
    gpu_monitor = GPUMonitorThread(
        device_index=0,
        interval=config["monitor_interval"],
        enable_sm_metrics=config["enable_sm_metrics"],
    )
    gpu_monitor.start()

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_metrics: List[Dict[str, Any]] = []

    global_step = 0
    start_time = time.time()

    try:
        for epoch in range(1, config["num_epochs"] + 1):
            t0 = time.time()

            train_loss, train_acc, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer,
                epoch, global_step, device,
                use_amp, amp_dtype, scaler,
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion,
                epoch, device, use_amp, amp_dtype,
            )

            scheduler.step()

            epoch_train_loss.append(train_loss)
            epoch_train_acc.append(train_acc)
            epoch_val_loss.append(val_loss)
            epoch_val_acc.append(val_acc)

            elapsed = time.time() - t0
            energy_J = gpu_monitor.total_energy_joules()

            print(
                f"[Epoch {epoch}/{config['num_epochs']}] "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} Acc {val_acc:.2f}% | "
                f"Time {elapsed:.1f}s | "
                f"Energy {energy_J/3600:.3f} Wh (GPU, aprox)"
            )

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
        gpu_monitor.stop()

    total_time = time.time() - start_time
    total_energy_J = gpu_monitor.total_energy_joules()
    print(
        f"Entrenamiento terminado en {total_time/3600:.2f} h, "
        f"energía total GPU ~ {total_energy_J/3600:.2f} Wh"
    )

    # Guardar modelo
    model_path = os.path.join(exp_dir, "convnext_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "config": config,
    }, model_path)
    print(f"Modelo guardado en: {model_path}")

    # CSV monitorización por segundo
    monitor_csv = os.path.join(exp_dir, "monitoring_per_second.csv")
    gpu_monitor.export_csv(monitor_csv)
    print(f"CSV de monitorización guardado en: {monitor_csv}")

    # CSV por época
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
        },
        "timestamp_start": datetime.datetime.fromtimestamp(shared_state.start_time).isoformat(),
        "timestamp_end": datetime.datetime.now().isoformat(),
    }

    summary_path = os.path.join(exp_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_summary, f, indent=4)
    print(f"Resumen de experimento guardado en: {summary_path}")


if __name__ == "__main__":
    main()
