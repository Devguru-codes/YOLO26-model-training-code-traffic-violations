# SadakSathi: Autonomous Traffic Violation (YOLO) Training Infrastructure

This repository contains the hyper-optimized training architecture utilized for the traffic-surveillance models inside the **SadakSathi** Complaint Management System. It is engineered to train Ultralytics YOLOv8 models autonomously using deep GPU hardware (e.g., NVIDIA Quadro).

### 📊 Dataset Link (Kaggle)
The isolated dataset associated with this training codebase is hosted securely on Kaggle due to its massive scale (~20,000 augmented images).
👉 **[Download the Dataset from Kaggle](https://www.kaggle.com/datasets/devgurucodes/trafffic-violations-triple-riding-no-helmet-plate/data)**

### ⚡ Classes Detected
1. `WithHelmet`
2. `WithoutHelmet`
3. `TripleRiding`
4. `Plate` (For downstream ALPR / Florence-2 Vision-Language extraction)

---

## 🚀 Execution & Usage

This suite automatically generates a clean Virtual Environment, installs exact CUDA/PyTorch dependencies, and triggers the YOLO pipeline. It records the hyperparameter tracking logs implicitly required for research paper metrics.

**Hardware Target:** Linux Virtual Machines / GPU Hubs

```bash
# Clone the Repository
git clone https://github.com/Devguru-codes/YOLO26-model-training-code-traffic-violations.git
cd YOLO26-model-training-code-traffic-violations

# Run the Automated GPU Setup
bash train_setup.sh
```

### 🧠 Core Subsystems
*   **`train_yolo.py`**: The raw training logic targeting `yolov8m.pt`. Configures `200 epochs` and handles the dynamic construction of `parameters_used.txt` for metrics logging.
*   **`requirements_train.txt`**: Minimal requirements optimized for CUDA environments (`torch`, `torchvision`).
*   **`train_setup.sh`**: Environment initialization and execution bash script.
