https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

pip install torch torchvision numpy matplotlib scikit-learn pynvml

python train_convnext_monitor.py \
  --data_dir /ruta/a/imagenet_10gb \
  --epochs 10 \
  --batch_size 128 \
  --lr 3e-4 \
  --seed 42
