from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")  # pretrained model

    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        name="yolo_custom"
    )

    print("✅ Training Completed!")

if __name__ == "__main__":
    train_model()