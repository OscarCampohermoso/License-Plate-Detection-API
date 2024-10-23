import cv2 as cv
import string
from easyocr import Reader
from models import np_model

# Mapping dictionaries for character conversion
# characters that can easily be confused can be
# verified by their location - an `O` in a place
# where a number is expected is probably a `0`
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5',
    'T': '7',
    'B': '8',
    'P': '9',
    'Z': '2',
    'D': '0',
    'Q': '9',
    'C': '6',
    'L': '1', 
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '8': 'B',
    '9': 'P',
    '2': 'Z',  
}

# Initialize OCR reader
reader = Reader(['es'])

def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    height = thresh.shape[0]
    width = thresh.shape[1]
    cropped_plate = thresh[int(height * 0.35):int(height * 0.85), int(width * 0.05):int(width * 0.95)]
    
    return cropped_plate

def clean_license_plate_text(text):
    valid_chars = '0123456789' + string.ascii_uppercase
    return ''.join(char for char in text if char in valid_chars)

def format_license(text):
    text = text.strip().upper()
    formatted_text = []

    for i in range(len(text) - 1, len(text) - 4, -1):
        if i >= 0 and text[i] in dict_int_to_char:
            formatted_text.append(dict_int_to_char[text[i]])
        elif i >= 0:
            formatted_text.append(text[i])

    for i in range(len(text) - 4, -1, -1):
        if text[i] in dict_char_to_int:
            formatted_text.append(dict_char_to_int[text[i]])
        else:
            formatted_text.append(text[i])

    formatted_text.reverse()
    return ''.join(formatted_text)

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = clean_license_plate_text(text.upper().replace(' ', ''))
        return format_license(text), score
    return None, None


def process_license_plate(image, roi, x1, y1, x2, y2, score):
    license_plate_data = None
    license_plates = np_model.predict(roi)[0]

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
            if x1 != 0 or y1 != 0 or x2 != image.shape[1] or y2 != image.shape[0]:
                cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv.rectangle(roi, (int(plate_x1), int(plate_y1)), (int(plate_x2), int(plate_y2)), (0, 255, 0), 2)
            cv.putText(roi, np_text, (int(plate_x1), int(plate_y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)
    
    return license_plate_data