import os
import json
from ultralytics import YOLO

def main():
    print("Initializing Autonomous YOLO Training Sequence for Quadro GPU...")
    
    # Define hyperparameter configuration strictly for Research Paper logging
    training_params = {
        "data": r"c:\Users\devgu\Downloads\helmet_violation_dataset\master_traffic_violation_dataset\data.yaml",  # Windows paths on the local machine
        "epochs": 200,
        "patience": 50,           # Early stopping: Halt if map50-95 doesn't improve for 50 epochs
        "imgsz": 640,             # Standard resolution for object tracking (Quadro handles this effortlessly)
        "batch": -1,              # Auto-batch configures the highest possible batch size based on Quadro VRAM
        "device": 0,              # Target NVIDIA GPU 0
        "optimizer": "auto",      # Let YOLO detect the best optimizer (AdamW usually)
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "augment": True,
        "mosaic": 1.0,            # High mosaic for complex clustered traffic scenes
        "mixup": 0.1,             # Mild mixup for overlapping vehicles
        "project": "Traffic_Violation_Model",
        "name": "yolov8m_helmet_triple_riding"
    }
    
    # 1. Save Hyperparameters locally for the Research Paper
    log_file_path = "parameters_used.txt"
    with open(log_file_path, "w") as f:
        f.write("==================================================\n")
        f.write("SADAKSATHI TRAFFIC VIOLATION YOLO HYPERPARAMETERS\n")
        f.write("==================================================\n\n")
        for key, value in training_params.items():
            f.write(f"{key}: {value}\n")
            
    print(f"[SUCCESS] Hyperparameters officially logged to {log_file_path} for your Research Paper.")
    
    # Notice: If deploying the bash script to a headless Linux VM, we re-parse the dataset path.
    # But since you mentioned running this, assuming Windows path format first:
    data_yaml = r"master_traffic_violation_dataset/data.yaml"
    if os.path.exists(data_yaml):
        training_params["data"] = data_yaml
    else:
        print(f"Warning: Ensure {training_params['data']} correctly maps to your dataset yaml!")
        
    print("\n--- INITIATING YOLO PIPELINE ---")
    
    # 2. Base Model Architecture
    # Using 'm' (Medium) architecture. It offers the best tradeoff of high-accuracy vs 40ms speed for Traffic CV. 
    model = YOLO('yolov8m.pt') 
    
    # 3. Model Training
    results = model.train(**training_params)

    print("\n[DONE] Model successfully trained across the specified epochs!")
    print(f"Final Weights Saved to the {training_params['project']}/{training_params['name']} directory.")

if __name__ == "__main__":
    main()
