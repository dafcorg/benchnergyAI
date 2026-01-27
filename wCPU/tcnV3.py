# train_convnext_monitor_ddp_lusitania3.py
# Script completo adaptado a LUSITANIA III (ppc64le) SIN torchvision:
# - Sustituye torchvision.datasets.ImageFolder -> Dataset propio (PIL)
# - Sustituye torchvision.transforms -> Albumentations + ToTensorV2
# - Sustituye torchvision.models.convnext_tiny -> timm.create_model("convnext_tiny")
# Mantiene: DDP multi-GPU, AMP, monitorización GPU/CPU, métricas por época, gráficas, modelo, config.json, summary.json
#
# Requisitos (conda-forge/pip):
#   pytorch, numpy, psutil, pynvml, matplotlib, scikit-learn, pillow, albumentations, opencv, timm
#
# Nota HPC:
# - Si no hay internet en los nodos, usa pretrained=False (por defecto) o carga pesos desde un checkpoint local.

import os
import time
import threading
import math
import csv
import json
import argparse
import random
import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from contextlib import nullcontext

import numpy as np
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

import pynvml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


# =========================
# Configuración por defecto
# =========================

CONFIG_DEFAULT = {
    "data_dir": "/home/david.cortesgomez/cis2/data/imagenet1k",  # Debe contener train/ y val/
    "log_dir": "/home/david.cortesgomez/cis2/results/convnext",  # (corregido: ruta absoluta)
    "batch_size": 128,                 # batch_size por GPU
    "num_workers": 8,
    "num_epochs": 20,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "monitor_interval": 1.0,           # segundos
    "seed": 1234,
    "device": "cuda",                  # En DDP se usa cuda:{LOCAL_RANK}; si no hay GPU -> cpu
    "precision": "fp32",               # "fp32", "fp16", "bf16"
    "backend": "nccl",                 # "nccl" (GPU) o "gloo"
    "seed_offset_per_rank": True,      # Semilla = seed + rank (reproducible y mejor para DDP)
    "enable_sm_metrics": False,        # Placeholder para CUPTI embebido (no implementado)
    "ncu_csv_path": None,              # Ruta a CSV de Nsight Compute (si se usa ncu)
    # Modelo
    "model_name": "convnext_tiny",     # timm model name
    "pretrained": False,               # True intenta descargar pesos (puede fallar sin internet)
    "pretrained_ckpt": None,           # Ruta a checkpoint .pth local (opcional) para cargar pesos
}


# =========================
# Utilidades DDP
# =========================

def is_distributed_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def ddp_is_main(rank: int, use_ddp: bool) -> bool:
    return (not use_ddp) or (rank == 0)

def ddp_barrier(use_ddp: bool):
    if use_ddp:
        dist.barrier()

def ddp_all_gather_object(obj: Any, use_ddp: bool) -> List[Any]:
    if not use_ddp:
        return [obj]
    out: List[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(out, obj)
    return out


# =========================
# Parsing CLI
# =========================

def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="ConvNeXt (timm) + monitorización + DDP multi-GPU + tracking JSON (sin torchvision)")

    parser.add_argument("--data_dir", type=str, default=CONFIG_DEFAULT["data_dir"])
    parser.add_argument("--log_dir", type=str, default=CONFIG_DEFAULT["log_dir"])
    parser.add_argument("--batch_size", type=int, default=CONFIG_DEFAULT["batch_size"])
    parser.add_argument("--num_workers", type=int, default=CONFIG_DEFAULT["num_workers"])
    parser.add_argument("--epochs", type=int, default=CONFIG_DEFAULT["num_epochs"])
    parser.add_argument("--lr", type=float, default=CONFIG_DEFAULT["lr"])
    parser.add_argument("--weight_decay", type=float, default=CONFIG_DEFAULT["weight_decay"])
    parser.add_argument("--monitor_interval", type=float, default=CONFIG_DEFAULT["monitor_interval"])
    parser.add_argument("--seed", type=int, default=CONFIG_DEFAULT["seed"])
    parser.add_argument("--device", type=str, default=CONFIG_DEFAULT["device"])
    parser.add_argument("--precision", type=str, default=CONFIG_DEFAULT["precision"], choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--backend", type=str, default=CONFIG_DEFAULT["backend"], choices=["nccl", "gloo"])
    parser.add_argument("--seed_offset_per_rank", action="store_true", default=CONFIG_DEFAULT["seed_offset_per_rank"])
    parser.add_argument("--enable_sm_metrics", action="store_true", help="Placeholder para CUPTI embebido (no implementado).")
    parser.add_argument("--ncu_csv_path", type=str, default=CONFIG_DEFAULT["ncu_csv_path"], help="CSV generado por ncu (Nsight Compute).")

    # Modelo timm / pesos
    parser.add_argument("--model_name", type=str, default=CONFIG_DEFAULT["model_name"], help="Nombre del modelo en timm, p.ej. convnext_tiny")
    parser.add_argument("--pretrained", action="store_true", default=CONFIG_DEFAULT["pretrained"], help="Usar pesos pretrained de timm (puede requerir internet/cache).")
    parser.add_argument("--pretrained_ckpt", type=str, default=CONFIG_DEFAULT["pretrained_ckpt"], help="Ruta a checkpoint .pth local para cargar pesos (opcional).")

    args = parser.parse_args()

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
        "backend": args.backend,
        "seed_offset_per_rank": args.seed_offset_per_rank,
        "enable_sm_metrics": args.enable_sm_metrics,
        "ncu_csv_path": args.ncu_csv_path,
        "model_name": args.model_name,
        "pretrained": bool(args.pretrained),
        "pretrained_ckpt": args.pretrained_ckpt,
    }
    return config


# =========================
# Reproducibilidad
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinismo (puede afectar rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Estado del algoritmo
# =========================

@dataclass
class AlgoState:
    timestamp: float = 0.0
    epoch: int = 0
    global_step: int = 0
    batch_in_epoch: int = 0
    phase: str = "idle"        # "train" / "val" / "idle"
    batch_loss: float = math.nan
    batch_acc: float = math.nan
    running_loss: float = math.nan
    running_acc: float = math.nan
    grad_norm: float = math.nan
    lr: float = math.nan


@dataclass
class MonitorRecord:
    # Tiempo
    timestamp: float
    rel_time_s: float

    # GPU (NVML)
    power_W: float
    util_gpu: float
    util_mem: float
    mem_used_MB: float
    temp_C: float

    # CPU nivel A (psutil)
    cpu_util_system_percent: float
    cpu_util_process_percent: float
    cpu_freq_mhz: float
    proc_ram_rss_mb: float
    proc_ram_vms_mb: float

    # Estado del algoritmo
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

    # Métricas SM avanzadas (placeholder)
    sm_occupancy: Optional[float] = None
    active_warps: Optional[float] = None
    inst_executed: Optional[float] = None
    stalls: Optional[float] = None


class SharedState:
    """Estado compartido entre el hilo de entrenamiento y el hilo de monitorización."""
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

    def snapshot(self) -> Tuple[AlgoState, float]:
        with self._lock:
            algo = AlgoState(**asdict(self._algo_state))
        rel_t = algo.timestamp - self._start_time
        return algo, rel_t

    @property
    def start_time(self) -> float:
        return self._start_time


shared_state = SharedState()


# =========================
# Monitor GPU + CPU + estado
# =========================

class GPUMonitorThread(threading.Thread):
    def __init__(self, device_index: int, interval: float, enable_sm_metrics: bool):
        super().__init__()
        self.device_index = device_index
        self.interval = interval
        self.enable_sm_metrics = enable_sm_metrics

        self._stop_flag = threading.Event()
        self.records: List[MonitorRecord] = []

        # NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

        # CPU (Nivel A) - proceso actual
        self.proc = psutil.Process(os.getpid())
        # Warm-up para que cpu_percent no devuelva 0
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)

    def _collect_sm_metrics(self):
        # Placeholder: aquí se integraría CUPTI embebido (no implementado)
        return None, None, None, None

    def run(self):
        while not self._stop_flag.is_set():
            now = time.time()
            algo_state, rel_t = shared_state.snapshot()

            # GPU - NVML
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_W = power_mW / 1000.0

            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used_MB = mem_info.used / (1024 ** 2)

            # CPU - Nivel A
            cpu_util_system = psutil.cpu_percent(interval=None)
            cpu_util_proc = self.proc.cpu_percent(interval=None)

            freq = psutil.cpu_freq()
            cpu_freq_mhz = freq.current if freq is not None else float("nan")

            mem = self.proc.memory_info()
            proc_ram_rss_mb = mem.rss / (1024 ** 2)
            proc_ram_vms_mb = mem.vms / (1024 ** 2)

            # SM metrics (placeholder)
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

                cpu_util_system_percent=cpu_util_system,
                cpu_util_process_percent=cpu_util_proc,
                cpu_freq_mhz=cpu_freq_mhz,
                proc_ram_rss_mb=proc_ram_rss_mb,
                proc_ram_vms_mb=proc_ram_vms_mb,

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
        # Integración trapezoidal de potencia vs tiempo
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
# Dataset (reemplazo de torchvision.datasets.ImageFolder)
# =========================

class SimpleImageFolder(Dataset):
    """
    Reemplazo mínimo de ImageFolder:
      root/
        class_a/*.jpg
        class_b/*.jpg
    """
    def __init__(self, root: str, transform=None, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
        self.root = root
        self.transform = transform
        self.exts = tuple(e.lower() for e in exts)

        self.classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for c in self.classes:
            cdir = os.path.join(root, c)
            for fn in os.listdir(cdir):
                if fn.lower().endswith(self.exts):
                    self.samples.append((os.path.join(cdir, fn), self.class_to_idx[c]))

        self.samples.sort(key=lambda x: x[0])

        if len(self.samples) == 0:
            raise RuntimeError(f"No encontré imágenes en {root} con extensiones {self.exts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = np.array(img)  # HWC uint8

        if self.transform is not None:
            out = self.transform(image=img)
            x = out["image"]  # torch.Tensor CHW
        else:
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return x, torch.tensor(y, dtype=torch.long)


def build_transforms(train: bool):
    # Normalización típica ImageNet
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        return A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.08, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    distributed: bool
) -> Tuple[DataLoader, DataLoader, int, Optional[DistributedSampler], Optional[DistributedSampler]]:

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    train_ds = SimpleImageFolder(train_path, transform=build_transforms(train=True))
    val_ds = SimpleImageFolder(val_path, transform=build_transforms(train=False))

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    num_classes = len(train_ds.classes)
    return train_loader, val_loader, num_classes, train_sampler, val_sampler


# =========================
# Modelo (reemplazo de torchvision.models)
# =========================

def load_pretrained_ckpt_if_any(model: nn.Module, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained_ckpt no existe: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    # Soporta:
    # - state_dict directo
    # - dict con claves típicas ("model", "state_dict", "model_state_dict")
    if isinstance(obj, dict):
        if "model_state_dict" in obj:
            sd = obj["model_state_dict"]
        elif "state_dict" in obj:
            sd = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            # puede ser que el dict ya sea el state_dict
            sd = obj
    else:
        sd = obj

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[CKPT] Missing keys (muestra 10): {missing[:10]}")
    if unexpected:
        print(f"[CKPT] Unexpected keys (muestra 10): {unexpected[:10]}")
    print(f"[CKPT] Cargado checkpoint: {ckpt_path}")


def get_model(num_classes: int, model_name: str, pretrained: bool, pretrained_ckpt: Optional[str]) -> nn.Module:
    # pretrained=True puede requerir internet o cache local de timm/huggingface
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # Si se proporciona ckpt local, lo aplicamos (por encima de pretrained si ambos están activos)
    if pretrained_ckpt:
        load_pretrained_ckpt_if_any(model, pretrained_ckpt)

    return model


# =========================
# Entrenamiento y evaluación
# =========================

def compute_grad_norm(model: nn.Module) -> float:
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    global_step: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    scaler: GradScaler,
    rank: int,
    use_ddp: bool
) -> Tuple[float, float, int]:
    model.train()
    shared_state.update_algo_state(phase="train", epoch=epoch)

    running_loss = 0.0
    running_correct = 0
    total = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

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
        bsz = targets.size(0)
        total += bsz
        running_loss += loss.item() * bsz
        running_correct += preds.eq(targets).sum().item()

        batch_loss = float(loss.item())
        batch_acc = float(preds.eq(targets).float().mean().item() * 100.0)
        running_loss_avg = float(running_loss / total)
        running_acc = float(running_correct / total * 100.0)
        lr = float(optimizer.param_groups[0]["lr"])

        global_step += 1

        shared_state.update_algo_state(
            epoch=epoch,
            global_step=global_step,
            batch_in_epoch=batch_idx,
            batch_loss=batch_loss,
            batch_acc=batch_acc,
            running_loss=running_loss_avg,
            running_acc=running_acc,
            grad_norm=float(grad_norm),
            lr=lr,
        )

        if ddp_is_main(rank, use_ddp) and (batch_idx + 1) % 10 == 0:
            print(
                f"[Train][rank0] Epoch {epoch} "
                f"Batch {batch_idx+1}/{len(loader)} "
                f"Loss {batch_loss:.4f} Acc {batch_acc:.2f}% "
                f"RunningLoss {running_loss_avg:.4f} RunningAcc {running_acc:.2f}% "
                f"GradNorm {grad_norm:.2f} LR {lr:.6f}"
            )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_correct / max(total, 1) * 100.0

    return float(epoch_loss), float(epoch_acc), global_step


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    epoch: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> Tuple[float, float]:
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
        bsz = targets.size(0)
        total += bsz
        running_loss += loss.item() * bsz
        running_correct += preds.eq(targets).sum().item()

        batch_loss = float(loss.item())
        batch_acc = float(preds.eq(targets).float().mean().item() * 100.0)

        shared_state.update_algo_state(
            batch_in_epoch=batch_idx,
            batch_loss=batch_loss,
            batch_acc=batch_acc,
            running_loss=float(running_loss / total),
            running_acc=float(running_correct / total * 100.0),
        )

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_correct / max(total, 1) * 100.0

    return float(epoch_loss), float(epoch_acc)


def ddp_reduce_mean(value: float, device: torch.device, use_ddp: bool) -> float:
    if not use_ddp:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / float(get_world_size())
    return float(t.item())


@torch.no_grad()
def predict_all_ddp(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    use_ddp: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    model.eval()

    local_preds: List[int] = []
    local_labels: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        if device.type == "cuda" and use_amp and amp_dtype is not None:
            ctx = autocast(device_type="cuda", dtype=amp_dtype)
        else:
            ctx = nullcontext()

        with ctx:
            outputs = model(images)

        _, preds = outputs.max(1)
        local_preds.extend(preds.detach().cpu().numpy().tolist())
        local_labels.extend(targets.detach().cpu().numpy().tolist())

    if not use_ddp:
        return np.array(local_labels, dtype=np.int64), np.array(local_preds, dtype=np.int64)

    # all_gather requiere tensores CUDA en NCCL
    local_n = len(local_labels)
    n_t = torch.tensor([local_n], device=device, dtype=torch.int64)

    sizes = [torch.zeros_like(n_t) for _ in range(get_world_size())]
    dist.all_gather(sizes, n_t)
    sizes_int = [int(s.item()) for s in sizes]
    max_n = max(sizes_int) if sizes_int else 0

    pad_value = -1
    lab_t = torch.full((max_n,), pad_value, device=device, dtype=torch.int64)
    pred_t = torch.full((max_n,), pad_value, device=device, dtype=torch.int64)

    if local_n > 0:
        lab_t[:local_n] = torch.tensor(local_labels, device=device, dtype=torch.int64)
        pred_t[:local_n] = torch.tensor(local_preds, device=device, dtype=torch.int64)

    gathered_lab = [torch.empty_like(lab_t) for _ in range(get_world_size())]
    gathered_pred = [torch.empty_like(pred_t) for _ in range(get_world_size())]
    dist.all_gather(gathered_lab, lab_t)
    dist.all_gather(gathered_pred, pred_t)

    if get_rank() != 0:
        return None, None

    all_labels = []
    all_preds = []
    for r in range(get_world_size()):
        n = sizes_int[r]
        if n > 0:
            all_labels.append(gathered_lab[r][:n].detach().cpu().numpy())
            all_preds.append(gathered_pred[r][:n].detach().cpu().numpy())

    if not all_labels:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    return y_true, y_pred


# =========================
# Nsight Compute CSV -> resumen SM
# =========================

def parse_ncu_csv(ncu_csv_path: Optional[str]) -> Optional[Dict[str, float]]:
    if ncu_csv_path is None:
        return None
    if not os.path.isfile(ncu_csv_path):
        return None

    occ_vals: List[float] = []
    warps_vals: List[float] = []
    inst_vals: List[float] = []

    occ_key = "sm__warps_active.avg.pct_of_peak_sustained_active"
    warps_key = "sm__warps_active.avg"
    inst_key = "sm__inst_executed.sum"

    with open(ncu_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if occ_key in row and row[occ_key] not in ("", "N/A"):
                    occ_vals.append(float(row[occ_key]))
                if warps_key in row and row[warps_key] not in ("", "N/A"):
                    warps_vals.append(float(row[warps_key]))
                if inst_key in row and row[inst_key] not in ("", "N/A"):
                    inst_vals.append(float(row[inst_key]))
            except ValueError:
                continue

    if not occ_vals and not warps_vals and not inst_vals:
        return None

    def safe_mean(a: List[float]) -> Optional[float]:
        return float(np.mean(a)) if a else None

    def safe_p95(a: List[float]) -> Optional[float]:
        return float(np.percentile(a, 95)) if a else None

    sm_profile = {
        "ncu_csv_path": os.path.abspath(ncu_csv_path),
        "sm_occupancy_avg_pct": safe_mean(occ_vals),
        "sm_occupancy_p95_pct": safe_p95(occ_vals),
        "warps_active_avg": safe_mean(warps_vals),
        "inst_executed_total": float(np.sum(inst_vals)) if inst_vals else None,
        "ncu_metrics_expected": [occ_key, warps_key, inst_key],
    }
    return sm_profile


# =========================
# Main
# =========================

def main():
    config = parse_args()

    use_ddp = is_distributed_env() and get_world_size() > 1
    rank = get_rank() if use_ddp else 0
    local_rank = get_local_rank() if use_ddp else 0
    world_size = get_world_size() if use_ddp else 1

    if use_ddp:
        dist.init_process_group(backend=config["backend"])
        torch.cuda.set_device(local_rank)

    # Device: 1 proceso por GPU (cuda:{LOCAL_RANK})
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available() and config["device"] == "cuda":
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Semilla por rank
    seed = int(config["seed"])
    if use_ddp and bool(config.get("seed_offset_per_rank", True)):
        seed = seed + rank
    set_seed(seed)

    # Directorio experimento
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if ddp_is_main(rank, use_ddp):
        exp_dir = os.path.join(config["log_dir"], f"exp_{timestamp_str}")
        os.makedirs(exp_dir, exist_ok=True)

        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    else:
        exp_dir = None
        config_path = None

    exp_dir_list = ddp_all_gather_object(exp_dir, use_ddp)
    exp_dir = exp_dir_list[0]
    ddp_barrier(use_ddp)

    if ddp_is_main(rank, use_ddp):
        print(f"DDP enabled: {use_ddp}")
        print(f"World size: {world_size}")
        print(f"Backend: {config['backend']}")
        print(f"Device: {device}")
        print(f"Precision: {config['precision']}")
        print(f"Experiment dir: {exp_dir}")
        print("Nota: batch_size es por GPU; batch global = batch_size * world_size.")
        print(f"Modelo timm: {config['model_name']} | pretrained={config['pretrained']} | ckpt={config['pretrained_ckpt']}")

    # AMP / dtype
    precision = config["precision"]
    if device.type == "cuda":
        if precision == "fp16":
            use_amp = True
            amp_dtype = torch.float16
            scaler = GradScaler(enabled=True)
        elif precision == "bf16":
            use_amp = True
            amp_dtype = torch.bfloat16
            scaler = GradScaler(enabled=False)
        else:
            use_amp = False
            amp_dtype = None
            scaler = GradScaler(enabled=False)
    else:
        use_amp = False
        amp_dtype = None
        scaler = GradScaler(enabled=False)

    # DataLoaders
    train_loader, val_loader, num_classes, train_sampler, val_sampler = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        distributed=use_ddp
    )

    # Modelo
    model = get_model(
        num_classes=num_classes,
        model_name=config["model_name"],
        pretrained=bool(config["pretrained"]),
        pretrained_ckpt=config.get("pretrained_ckpt"),
    ).to(device)

    # Wrap DDP
    if use_ddp and device.type == "cuda":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["num_epochs"]))

    # Monitor
    device_index_for_nvml = local_rank if device.type == "cuda" else 0
    gpu_monitor = GPUMonitorThread(
        device_index=device_index_for_nvml,
        interval=float(config["monitor_interval"]),
        enable_sm_metrics=bool(config["enable_sm_metrics"])
    )
    gpu_monitor.start()

    epoch_train_loss: List[float] = []
    epoch_train_acc: List[float] = []
    epoch_val_loss: List[float] = []
    epoch_val_acc: List[float] = []
    epoch_metrics: List[Dict[str, Any]] = []

    global_step = 0
    start_time = time.time()

    try:
        for epoch in range(1, int(config["num_epochs"]) + 1):
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            t0 = time.time()

            train_loss, train_acc, global_step = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                rank=rank,
                use_ddp=use_ddp
            )

            val_loss, val_acc = validate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                epoch=epoch,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )

            # Reducir métricas por época
            train_loss_g = ddp_reduce_mean(train_loss, device, use_ddp)
            train_acc_g = ddp_reduce_mean(train_acc, device, use_ddp)
            val_loss_g = ddp_reduce_mean(val_loss, device, use_ddp)
            val_acc_g = ddp_reduce_mean(val_acc, device, use_ddp)

            scheduler.step()

            epoch_train_loss.append(train_loss_g)
            epoch_train_acc.append(train_acc_g)
            epoch_val_loss.append(val_loss_g)
            epoch_val_acc.append(val_acc_g)

            elapsed = time.time() - t0
            energy_J_rank = gpu_monitor.total_energy_joules()

            if use_ddp:
                e_t = torch.tensor([energy_J_rank], device=device, dtype=torch.float64)
                dist.all_reduce(e_t, op=dist.ReduceOp.SUM)
                energy_J_total = float(e_t.item())
            else:
                energy_J_total = float(energy_J_rank)

            if ddp_is_main(rank, use_ddp):
                print(
                    f"[Epoch {epoch}/{config['num_epochs']}] "
                    f"Train Loss {train_loss_g:.4f} Acc {train_acc_g:.2f}% | "
                    f"Val Loss {val_loss_g:.4f} Acc {val_acc_g:.2f}% | "
                    f"Time {elapsed:.1f}s | "
                    f"EnergyGPU {energy_J_total/3600.0:.3f} Wh (aprox total GPUs)"
                )

                epoch_metrics.append({
                    "epoch": epoch,
                    "train_loss": train_loss_g,
                    "train_acc": train_acc_g,
                    "val_loss": val_loss_g,
                    "val_acc": val_acc_g,
                    "epoch_time_s": float(elapsed),
                    "energy_gpu_Wh_total_cumulative": float(energy_J_total / 3600.0),
                })

    finally:
        gpu_monitor.stop()

    total_time = time.time() - start_time
    energy_J_rank = gpu_monitor.total_energy_joules()

    if use_ddp and device.type == "cuda":
        e_t = torch.tensor([energy_J_rank], device=device, dtype=torch.float64)
        dist.all_reduce(e_t, op=dist.ReduceOp.SUM)
        total_energy_J = float(e_t.item())
    else:
        total_energy_J = float(energy_J_rank)

    # Export CSV monitor por rank
    monitor_csv_rank = os.path.join(exp_dir, f"monitoring_per_second_rank{rank}.csv")
    gpu_monitor.export_csv(monitor_csv_rank)

    # Parsear ncu CSV por rank (si existe)
    sm_profile_rank = parse_ncu_csv(config.get("ncu_csv_path"))
    sm_profiles_all = ddp_all_gather_object(sm_profile_rank, use_ddp)

    if ddp_is_main(rank, use_ddp):
        model_to_save = model.module if (use_ddp and isinstance(model, DDP)) else model
        model_path = os.path.join(exp_dir, "convnext_model.pth")
        torch.save({
            "model_state_dict": model_to_save.state_dict(),
            "num_classes": num_classes,
            "config": config,
            "world_size": world_size,
            "batch_size_per_gpu": int(config["batch_size"]),
            "global_batch_size": int(config["batch_size"]) * int(world_size),
        }, model_path)

        # CSV por época
        epoch_csv = os.path.join(exp_dir, "training_epochs.csv")
        with open(epoch_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
            for e, tl, ta, vl, va in zip(
                range(1, int(config["num_epochs"]) + 1),
                epoch_train_loss,
                epoch_train_acc,
                epoch_val_loss,
                epoch_val_acc
            ):
                writer.writerow([e, tl, ta, vl, va])

        # Predicciones globales (val) para confusion matrix y F1
        y_true, y_pred = predict_all_ddp(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_ddp=use_ddp
        )

        if y_true is None or y_pred is None:
            y_true = np.array([], dtype=np.int64)
            y_pred = np.array([], dtype=np.int64)

        cm = confusion_matrix(y_true, y_pred) if y_true.size > 0 else np.zeros((num_classes, num_classes), dtype=np.int64)
        f1_macro = float(f1_score(y_true, y_pred, average="macro")) if y_true.size > 0 else float("nan")

        print("Matriz de confusión (val, global):")
        print(cm)
        print(f"F1 macro (val, global): {f1_macro:.6f}")

        # Gráfica loss/accuracy
        epochs = list(range(1, int(config["num_epochs"]) + 1))
        loss_acc_fig = os.path.join(exp_dir, "loss_accuracy.png")

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

        plt.tight_layout()
        plt.savefig(loss_acc_fig, dpi=150)
        plt.close()

        # Gráfica monitorización rank0
        monitor_fig_rank0 = os.path.join(exp_dir, "monitoring_raw_rank0.png")

        times = []
        power = []
        util_gpu = []
        mem_used = []
        temp = []
        cpu_sys = []
        cpu_proc = []

        with open(os.path.join(exp_dir, "monitoring_per_second_rank0.csv"), "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row["rel_time_s"]))
                power.append(float(row["power_W"]))
                util_gpu.append(float(row["util_gpu"]))
                mem_used.append(float(row["mem_used_MB"]))
                temp.append(float(row["temp_C"]))
                cpu_sys.append(float(row["cpu_util_system_percent"]))
                cpu_proc.append(float(row["cpu_util_process_percent"]))

        plt.figure(figsize=(11, 7))
        plt.plot(times, power, label="GPU Power (W)")
        plt.plot(times, util_gpu, label="GPU Util (%)")
        plt.plot(times, mem_used, label="GPU Mem Used (MB)")
        plt.plot(times, temp, label="GPU Temp (C)")
        plt.plot(times, cpu_sys, label="CPU System Util (%)")
        plt.plot(times, cpu_proc, label="CPU Process Util (%)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Valor")
        plt.title("Monitorización por segundo (rank0, valores crudos)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(monitor_fig_rank0, dpi=150)
        plt.close()

        monitor_csvs = [os.path.join(exp_dir, f"monitoring_per_second_rank{r}.csv") for r in range(world_size)]

        summary_path = os.path.join(exp_dir, "experiment_summary.json")
        exp_summary = {
            "config": config,
            "ddp": {
                "enabled": bool(use_ddp),
                "world_size": int(world_size),
                "backend": config["backend"],
                "batch_size_per_gpu": int(config["batch_size"]),
                "global_batch_size": int(config["batch_size"]) * int(world_size),
            },
            "num_classes": int(num_classes),
            "total_time_s": float(total_time),
            "total_energy_gpu_J_approx": float(total_energy_J),
            "total_energy_gpu_Wh_approx": float(total_energy_J / 3600.0),
            "final_f1_macro": float(f1_macro),
            "final_confusion_matrix": cm.tolist(),
            "epoch_metrics": epoch_metrics,
            "paths": {
                "experiment_dir": exp_dir,
                "config_json": os.path.join(exp_dir, "config.json"),
                "model": model_path,
                "epoch_csv": epoch_csv,
                "loss_acc_fig": loss_acc_fig,
                "monitor_fig_rank0": monitor_fig_rank0,
                "monitor_csv_per_rank": monitor_csvs,
                "ncu_csv_path_per_rank": [config.get("ncu_csv_path") for _ in range(world_size)],
            },
            "sm_profile_per_rank": sm_profiles_all,
            "timestamp_start": datetime.datetime.fromtimestamp(shared_state.start_time).isoformat(),
            "timestamp_end": datetime.datetime.now().isoformat(),
        }

        with open(summary_path, "w") as f:
            json.dump(exp_summary, f, indent=4)

        print(f"Resumen guardado en: {summary_path}")
        print(f"Modelo guardado en: {model_path}")
        print("CSV de monitorización por rank:")
        for p in monitor_csvs:
            print(f"  {p}")

    ddp_barrier(use_ddp)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
