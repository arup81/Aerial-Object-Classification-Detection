from ultralytics import YOLO

def predict_image():
    model = YOLO("runs/detect/yolo_custom7/weights/best.pt")

    results = model("test.jpg", show=True)

    results[0].save("output.jpg")

    print("✅ Prediction saved as output.jpg")

if __name__ == "__main__":
    predict_image()