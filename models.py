from ultralytics import YOLO

# Pre-trained YOLO models
coco_model = YOLO('yolov8s.pt')  # Car detection model
np_model = YOLO('./runs/detect/train/weights/best.pt')  # License plate detection model
