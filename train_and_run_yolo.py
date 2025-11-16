import os
from ultralytics import YOLO
import argparse

# --- Dataset path ---
DATASET_DIR = "Tree-Detection-2"  # folder with train/test and data.yaml
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
DEFAULT_MODEL = "yolov8n.pt"      # pre-trained YOLOv8 nano
PREDICT_SOURCE = os.path.join(DATASET_DIR, "test")  # test folder

def train_model():
    print(f"🌱 Training YOLOv8 model on dataset: {DATA_YAML}")
    model = YOLO(DEFAULT_MODEL)
    model.train(data=DATA_YAML, epochs=50, imgsz=640)
    print("✅ Training complete! Model saved under 'runs/detect/train/weights/best.pt'")

def run_inference(source=PREDICT_SOURCE):
    # Use the latest trained model
    weights_path = "runs/detect/train4/weights/best.pt"
    if not os.path.exists(weights_path):
        print("❌ No trained model found. Please run training first.")
        return

    print(f"🔍 Running inference on: {source}")
    model = YOLO(weights_path)
    model.predict(source=source, conf=0.25, save=True)
    print("✅ Prediction complete! Check 'runs/detect/predict/' for results.")

def main():
    parser = argparse.ArgumentParser(description="Train or run YOLO tree detector")
    parser.add_argument("mode", choices=["train", "predict"], help="Mode: train or predict")
    args = parser.parse_args()

    # Ensure we are in repo root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        run_inference()

if __name__ == "__main__":
    main()
