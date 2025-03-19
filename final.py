from ultralytics import YOLO

# Initialize YOLOv3 model
model = YOLO("yolov3.pt")  # Uses YOLOv3 architecture

# Train the model
model.train(
    data="C:/Users/Sujana/Desktop/PROJECT NEW 10/YOLOv3_Custom/data.yaml",  # Path to your dataset config
    epochs=100,       # Increase for better results
    batch=16,         # Adjust based on GPU memory
    imgsz=640,        # Resize images
    device="CPU",    # Use 'cuda' for GPU, 'cpu' for CPU
    project="C:/Users/Sujana/Desktop/PROJECT NEW 10/YOLOv3_Custom/",  # Set custom save location
    name="yolov3_train"  # Name of training run
)
