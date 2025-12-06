import os
from ultralytics import YOLO

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------
DATASET_DIR = "Tree-Segmentation-2"       # your local dataset folder
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

# YOLO segmentation pretrained model (choose one)
MODEL_WEIGHTS = "yolov8n-seg.pt"     # nano (fast, good for small datasets)
# MODEL_WEIGHTS = "yolov8s-seg.pt"   # small (higher accuracy)
# MODEL_WEIGHTS = "yolov8m-seg.pt"   # medium
# MODEL_WEIGHTS = "yolov8l-seg.pt"   # large

PREDICT_SOURCE = os.path.join(DATASET_DIR, "test")  # folder of test images

# -----------------------------------------------------
# TRAINING FUNCTION
# -----------------------------------------------------
def train_model():
    print(f"🌲 Training YOLOv8 SEGMENTATION model")
    print(f"📁 Dataset: {DATA_YAML}")

    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=4,
        device="cuda"  # or "cpu"
    )

    print("✅ Training complete!")
    print("📦 Model saved under: runs/segment/train*/weights/best.pt")


# -----------------------------------------------------
# INFERENCE FUNCTION
# -----------------------------------------------------
def run_inference(source=PREDICT_SOURCE):
    # Autodetect latest training run
    runs = [d for d in os.listdir("runs/segment") if d.startswith("train")]
    if not runs:
        print("❌ No segmentation model found. Train first.")
        return

    latest_run = sorted(runs)[-1]
    weights_path = os.path.join("runs/segment", latest_run, "weights", "best.pt")

    print(f"🔍 Running inference with: {weights_path}")
    model = YOLO(weights_path)

    model.predict(
        source=source,
        conf=0.25,
        save=True,
        device="cuda"  # or "cpu"
    )

    print("✅ Predictions saved under: runs/segment/predict*/")


# -----------------------------------------------------
# MAIN EXECUTION (simple CLI)
# -----------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"], help="train or predict")
    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        run_inference()

