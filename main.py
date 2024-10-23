from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from models import coco_model
from image_processing import preprocess_image, read_license_plate, process_license_plate
import numpy as np
from PIL import Image
import io
import cv2 as cv

app = FastAPI()


@app.post("/detect-license-plate/")
async def detect_license_plate(file: UploadFile = File(...)):
    image = np.array(Image.open(io.BytesIO(await file.read())))

    coco_model.callbacks = {}  # Remove tracking callbacks
    vehicles = [2, 3, 5, 7]  # Vehicle class IDs (cars, trucks)

    detections = coco_model.predict(image)[0]  # Make sure this is just detection, no tracking
    license_plate_data = None
    vehicle_bounding_boxes = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        print(score, class_id)
        if int(class_id) in vehicles and score > 0.5:
            vehicle_bounding_boxes.append([x1, y1, x2, y2, None, score])  # track_id is None
            # Crop the vehicle's region of interest (ROI)
            roi = image[int(y1):int(y2), int(x1):int(x2)]

            # Process license plate in the vehicle's ROI
            license_plate_data = process_license_plate(image, roi, x1, y1, x2, y2, score) or license_plate_data

    # If no vehicles were detected, try to detect license plates in the whole image
    if not vehicle_bounding_boxes:
        print("No vehicles detected, trying to detect license plates in the whole image.")
        license_plate_data = process_license_plate(image, image, 0, 0, image.shape[1], image.shape[0], 1.0)

    _, img_encoded = cv.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    # Return the image and license plate data if found
    return {
        "image": StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg"),
        "license_plate_data": license_plate_data
    }