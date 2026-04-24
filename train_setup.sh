#!/bin/bash
set -e

echo "=========================================================="
echo "Initializing Environment for YOLO Traffic Violation Model"
echo "Target Hardware: NVIDIA Quadro 4000 (CUDA)"
echo "=========================================================="

echo "[1/3] Creating isolated Python venv 'training_env'..."
python3 -m venv training_env
source training_env/bin/activate

echo "[2/3] Installing Dependencies and PyTorch CUDA Modules..."
pip install --upgrade pip
pip install -r requirements_train.txt

echo "[3/3] Environment Locked. Triggering YOLOv8 Pipeline..."
python train_yolo.py

echo "=========================================================="
echo "Execution Process Finished. Check output directories."
echo "=========================================================="
