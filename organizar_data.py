import os
import random
import shutil
from pathlib import Path

# Ruta a la carpeta que contiene las 1000 clases
# Ejemplo: "D:/datasets/imagenet1k" o "/home/usuario/datos/imagenet1k"
ROOT_DIR = Path("C:/Users/Fer03/Documents/Code/imageNet1k/imagenet1k") # Cambiar

# Porcentaje de imágenes que irán a train
TRAIN_RATIO = 0.8  # 80% train, 10% val

# Semilla para hacer el split reproducible
SEED = 1234

# Si True -> mueve los archivos (no duplica). Si False -> copia los archivos.
MOVE_FILES = True


def organizar_imagenet_stable(
    root_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 1234,
    move_files: bool = True,
):
    random.seed(seed)

    train_root = root_dir / "train"
    val_root = root_dir / "val"

    train_root.mkdir(exist_ok=True)
    val_root.mkdir(exist_ok=True)

    class_dirs = [
        d for d in root_dir.iterdir()
        if d.is_dir() and d.name not in ["train", "val"]
    ]

    print(f"Encontradas {len(class_dirs)} clases.")

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        print(f"\nProcesando clase: {class_name}")
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file()
        ]

        if not image_files:
            print(f"  [AVISO] Clase {class_name} sin imágenes, se salta.")
            continue
        random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = n_total - n_train

        train_files = image_files[:n_train]
        val_files = image_files[n_train:]

        print(f"  Total imágenes: {n_total}")
        print(f"  -> Train: {n_train}")
        print(f"  -> Val:   {n_val}")

        train_class_dir = train_root / class_name
        val_class_dir = val_root / class_name
        train_class_dir.mkdir(exist_ok=True, parents=True)
        val_class_dir.mkdir(exist_ok=True, parents=True)

        op = shutil.move if move_files else shutil.copy2

        for src in train_files:
            dst = train_class_dir / src.name
            op(str(src), str(dst))

        for src in val_files:
            dst = val_class_dir / src.name
            op(str(src), str(dst))

        if move_files:
            try:
                class_dir.rmdir()
                print(f"  Carpeta original vacía eliminada: {class_dir}")
            except OSError:
                print(f"  Carpeta original {class_dir} no está vacía, no se elimina.")

    print("\nFinal final no va más.")
    print(f"Train en: {train_root}")
    print(f"Val   en: {val_root}")


if __name__ == "__main__":
    print(f"Usando ROOT_DIR = {ROOT_DIR.resolve()}")
    organizar_imagenet_stable(
        ROOT_DIR,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        move_files=MOVE_FILES,
    )
