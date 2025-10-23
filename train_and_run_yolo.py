import os
from ultralytics import YOLO
import argparse
from glob import glob

# --- Constants / Defaults ---
DATA_PATH = "dataset/data.yaml"
DEFAULT_MODEL = "yolov8n.pt"
TRAIN_DIR = "runs/detect/train"
PREDICT_SOURCE = "test_images"  # default folder for prediction


def get_latest_model():
    """Find the most recent best.pt model in runs/detect/train directories."""
    paths = sorted(glob(os.path.join(TRAIN_DIR, "*/weights/best.pt")), key=os.path.getmtime, reverse=True)
    return paths[0] if paths else None


def train_model():
    print("🌱 Starting YOLO training...")
    model = YOLO(DEFAULT_MODEL)
    model.train(data=DATA_PATH, epochs=50, imgsz=640)
    print("\n✅ Training complete! Model saved under 'runs/detect/train/weights/best.pt'")


def run_inference(source):
    model_path = get_latest_model()
    if not model_path:
        print("❌ No trained model found. Please run training first.")
        return

    print(f"🔍 Using model: {model_path}")
    print(f"📸 Detecting trees in: {source}")
    model = YOLO(model_path)
    results = model.predict(source=source, conf=0.25, save=True)
    print("\n✅ Detection complete! Results in 'runs/detect/predict/'.")


def main():
    parser = argparse.ArgumentParser(description="Train or run YOLO tree detector")
    parser.add_argument("mode", choices=["train", "predict"], help="Choose train or predict mode")
    parser.add_argument("--source", default=PREDICT_SOURCE, help="Image or folder for prediction (default: test_images)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        run_inference(args.source)


if __name__ == "__main__":
    main()
