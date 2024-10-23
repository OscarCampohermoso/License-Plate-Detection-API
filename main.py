from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from models import coco_model, np_model
from image_processing import preprocess_image, read_license_plate, process_license_plate
import numpy as np
from PIL import Image
import io
import cv2 as cv

app = FastAPI()

@app.post("/detect-license-plate/")
async def detect_license_plate(file: UploadFile = File(...)):
    # Check the file extension
    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image = np.array(Image.open(io.BytesIO(await file.read())))
    else:
        return {"error": "Unsupported file type. Please upload a JPEG or PNG image."}

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

            # License plate detector for the region of interest
            license_plates = np_model.predict(roi)[0]

            # Process license plate
            for license_plate in license_plates.boxes.data.tolist():
                plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

                # Crop the plate from the region of interest
                plate = roi[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]

                # Put filters on the plate
                new_image = preprocess_image(plate)

                # OCR to read the license plate text
                np_text, np_score = read_license_plate(new_image)
                print(np_text, np_score)
                # If plate is readable, store results
                if np_text is not None:
                    license_plate_data = {
                        'car_bbox': [x1, y1, x2, y2] if x1 != 0 or y1 != 0 or x2 != image.shape[1] or y2 != image.shape[0] else None,
                        'car_bbox_score': score if x1 != 0 or y1 != 0 or x2 != image.shape[1] or y2 != image.shape[0] else None,
                        'plate_bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                        'plate_bbox_score': plate_score,
                        'plate_number': np_text,
                        'plate_text_score': np_score
                    }

                    # Draw vehicle and license plate bounding boxes on the image
                    cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 2)
                    cv.putText(roi, np_text, (int(plate_x1), int(plate_y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)

    # If no vehicles were detected, try to detect license plates in the whole image
    if not vehicle_bounding_boxes:
        print("No vehicles detected, trying to detect license plates in the whole image.")
        license_plates = np_model.predict(image)[0]
        for license_plate in license_plates.boxes.data.tolist():
            plate_x1, plate_y1, plate_x2, plate_y2, plate_score, _ = license_plate

            # Crop the plate from the whole image
            plate = image[int(plate_y1):int(plate_y2), int(plate_x1):int(plate_x2)]

            # Put filters on the plate
            new_image = preprocess_image(plate)

            # OCR to read the license plate text
            np_text, np_score = read_license_plate(new_image)
            # show cropped plate

            print(np_text, np_score)
            # If plate is readable, store results
            if np_text is not None:
                license_plate_data = {
                    'car_bbox': [x1, y1, x2, y2] if x1 != 0 or y1 != 0 or x2 != image.shape[1] or y2 != image.shape[0] else None,
                    'car_bbox_score': score if x1 != 0 or y1 != 0 or x2 != image.shape[1] or y2 != image.shape[0] else None,
                    'plate_bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                    'plate_bbox_score': plate_score,
                    'plate_number': np_text,
                    'plate_text_score': np_score
                }

                # Draw license plate bounding box on the image
                cv.rectangle(image, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 2)
                cv.putText(image, np_text, (int(plate_x1), int(plate_y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 2)

    _, img_encoded = cv.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    # Return the image and license plate data if found
    return {
        "image": StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg"),
        "license_plate_data": license_plate_data
    }