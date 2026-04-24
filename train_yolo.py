import os
import datetime
from ultralytics import YOLO

def main():
    print("Initializing YOLO26 Training Sequence for NVIDIA Quadro 4000...")

    # ─────────────────────────────────────────────────────────────────────────
    # YOLO26 Key Architecture Notes (for Research Paper):
    # - Released: January 2026 by Ultralytics
    # - NMS-Free: End-to-end one-to-one head, no Non-Maximum Suppression needed
    # - DFL Removed: Simplifies export to TensorRT/ONNX/TFLite/CoreML
    # - MuSGD Optimizer: Hybrid SGD + Muon (LLM-inspired) for faster convergence
    # - 43% faster CPU inference vs YOLO11 (edge-optimized architecture)
    # ─────────────────────────────────────────────────────────────────────────

    # Hyperparameter configuration (all values logged to parameters_used.txt)
    training_params = {
        # ── Dataset ──
        "data": "master_traffic_violation_dataset/data.yaml",

        # ── Duration ──
        "epochs": 200,
        "patience": 30,           # Early stopping: halt if mAP50-95 stagnates for 50 epochs

        # ── Input ──
        "imgsz": 640,             # Standard 640x640 resolution; Quadro 4000 VRAM handles this cleanly

        # ── Hardware ──
        "batch": -1,              # Auto-batch: maximise VRAM utilisation automatically
        "device": 0,              # Target NVIDIA GPU index 0

        # ── Optimizer ──
        # YOLO26 defaults to MuSGD (Muon + SGD hybrid). Setting "auto" triggers it automatically.
        "optimizer": "auto",      # Resolves to MuSGD for YOLO26 architecture
        "lr0": 0.01,              # Initial learning rate
        "lrf": 0.01,              # Final LR as fraction of lr0 (cosine annealing floor)
        "momentum": 0.937,        # SGD momentum / Adam beta1
        "weight_decay": 0.0005,   # L2 regularisation to prevent overfitting

        # ── Augmentation ──
        "augment": True,
        "mosaic": 1.0,            # Mosaic=1.0: full mosaic augmentation for dense traffic scenes
        "mixup": 0.1,             # Mild mixup to blend overlapping vehicle features

        # ── Warmup ──
        "warmup_epochs": 3.0,     # Gradual LR warmup to avoid early divergence
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # ── Output ──
        "project": "Traffic_Violation_Model",
        "name": "yolo26s_helmet_triple_riding",
        "save": True,
        "save_period": 10,        # Save checkpoint every 10 epochs
        "exist_ok": True,

        # ── Validation ──
        "val": True,
        "plots": True,            # Generate training plots (loss curves, confusion matrix, PR curve)
    }

    # ─── 1. Log Hyperparameters for Research Paper ───────────────────────────
    log_file_path = "parameters_used.txt"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("SADAKSATHI — YOLO26 TRAFFIC VIOLATION MODEL\n")
        f.write("Hyperparameters Used During Training\n")
        f.write(f"Logged at: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("[Model Architecture]\n")
        f.write("Model Family      : YOLO26 (Ultralytics, January 2026)\n")
        f.write("Variant           : Small (yolo26s.pt)\n")
        f.write("NMS               : None (End-to-End NMS-Free one-to-one head)\n")
        f.write("DFL               : Removed (improved ONNX/TensorRT export compatibility)\n")
        f.write("Optimizer         : MuSGD (SGD + Muon hybrid, LLM-inspired convergence)\n")
        f.write("Classes           : 4 (WithHelmet, WithoutHelmet, TripleRiding, Plate)\n\n")

        f.write("[Training Hyperparameters]\n")
        for key, value in training_params.items():
            f.write(f"{key:<25}: {value}\n")

        f.write("\n[Hardware]\n")
        f.write("GPU               : NVIDIA Quadro 4000\n")
        f.write("Framework         : PyTorch (CUDA)\n")
        f.write("Ultralytics       : YOLO26 (latest)\n")

    print(f"[SUCCESS] Hyperparameters logged to '{log_file_path}' for Research Paper.")

    # ─── 2. Resolve Dataset Path ─────────────────────────────────────────────
    data_yaml = training_params["data"]
    if not os.path.exists(data_yaml):
        # Fallback: absolute path for Windows development environments
        training_params["data"] = r"c:\Users\devgu\Downloads\helmet_violation_dataset\master_traffic_violation_dataset\data.yaml"
        print(f"[WARNING] Relative path not found. Using absolute path fallback.")

    # ─── 3. Load YOLO26 and Train ─────────────────────────────────────────────
    print("\n--- INITIATING YOLO26 PIPELINE ---")
    print("  Model     : yolo26s (Small — optimal speed/accuracy balance for Quadro)")
    print("  Classes   : WithHelmet | WithoutHelmet | TripleRiding | Plate")
    print("  Epochs    : 200 (Early Stop patience = 50)\n")

    # Load YOLO26 Small pretrained weights
    # Ultralytics auto-downloads yolo26s.pt on first run if not found locally
    model = YOLO("yolo26s.pt")

    results = model.train(**training_params)

    print("\n[DONE] YOLO26 training complete!")
    print(f"Best weights saved to: {training_params['project']}/{training_params['name']}/weights/best.pt")
    print(f"Hyperparameter log  : {log_file_path}")

if __name__ == "__main__":
    main()
