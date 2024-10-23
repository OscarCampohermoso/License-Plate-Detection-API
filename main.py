from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from models import coco_model, np_model
from image_processing import preprocess_image, read_license_plate, process_license_plate
import numpy as np
from PIL import Image
import io
import cv2 as cv
import os
import time
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Directorio donde se guardarán las imágenes procesadas
IMAGE_SAVE_PATH = "processed_images/"

# Crear el directorio si no existe
if not os.path.exists(IMAGE_SAVE_PATH):
    os.makedirs(IMAGE_SAVE_PATH)

# Montar el directorio de archivos estáticos para poder acceder a las imágenes
app.mount("/processed_images", StaticFiles(directory=IMAGE_SAVE_PATH), name="processed_images")

@app.post("/detect-license-plate/")
async def detect_license_plate(file: UploadFile = File(...)):
    # Check the file extension
    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image = np.array(Image.open(io.BytesIO(await file.read())))
    else:
        return {"error": "Unsupported file type. Please upload a JPEG or PNG image."}

    # Desactivar callbacks de tracking
    coco_model.callbacks = {}

    # IDs de clase para vehículos (coche, camión, etc.)
    vehicles = [2, 3, 5, 7]  

    # Comenzar el proceso y medir tiempos
    start_time = time.time()
    detections = coco_model.predict(image)[0]
    preprocess_time = (time.time() - start_time) * 1000  # Tiempo en milisegundos

    license_plate_data = None
    vehicle_bounding_boxes = []
    detection_results = []

    # Procesar las detecciones
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        # Guardar la información de la detección
        detection_results.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": score,
            "class_id": int(class_id)
        })

        # Verificar si se detectó un vehículo con una confianza mayor a 0.5
        if int(class_id) in vehicles and score > 0.5:
            vehicle_bounding_boxes.append([x1, y1, x2, y2, None, score])
            
            # Recortar la región de interés (ROI) del vehículo
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

    # Si no se detectaron vehículos, intentar detectar matrículas en toda la imagen
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


    # Guardar la imagen procesada en el disco
    image_filename = f"output_{int(time.time())}.jpg"
    image_path = os.path.join(IMAGE_SAVE_PATH, image_filename)
    cv.imwrite(image_path, image)

    # Capturar el tiempo de inferencia y postprocesamiento
    inference_time = (time.time() - start_time - preprocess_time / 1000) * 1000
    postprocess_time = 0.7  # Ejemplo de tiempo de postprocesamiento, ajustable según tu modelo

    # Generar la URL de la imagen procesada para acceso público
    host = os.getenv('HOST', 'localhost')
    port = os.getenv('PORT', '8000')
    image_url = f"http://{host}:{port}/processed_images/{image_filename}"

    # Preparar la respuesta JSON con todos los datos
    response_data = {
        "detections": detection_results,  # Lista de detecciones con coordenadas y clasificaciones
        "license_plate_data": license_plate_data,  # Datos de matrícula detectados (si aplica)
        "image_url": image_url,  # URL de la imagen procesada
        "speed": {
            "preprocess_time_ms": preprocess_time,
            "inference_time_ms": inference_time,
            "postprocess_time_ms": postprocess_time
        }
    }

    # Devolver la respuesta en formato JSON
    return JSONResponse(content=response_data)
