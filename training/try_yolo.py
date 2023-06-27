from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='/media/n31v/data/datasets/yolo_minerals/yolo_minerals.yaml',
    epochs=300,
    device=0,
)
