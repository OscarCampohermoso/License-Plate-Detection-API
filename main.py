from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from models import coco_model
from image_processing import process_license_plate
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
    # Cargar la imagen desde el archivo subido
    image = np.array(Image.open(io.BytesIO(await file.read())))

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

            # Procesar la matrícula en el ROI del vehículo
            license_plate_data = process_license_plate(image, roi, x1, y1, x2, y2, score) or license_plate_data

    # Si no se detectaron vehículos, intentar detectar matrículas en toda la imagen
    if not vehicle_bounding_boxes:
        print("No se detectaron vehículos, intentando detectar matrículas en toda la imagen.")
        license_plate_data = process_license_plate(image, image, 0, 0, image.shape[1], image.shape[0], 1.0)

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
