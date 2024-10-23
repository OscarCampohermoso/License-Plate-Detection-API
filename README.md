### README

# License Plate Detection API

This FastAPI application detects license plates in images. It uses a pre-trained COCO model to detect vehicles and a custom model to detect license plates within those vehicles.

## Setup

### Create a Conda Environment

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) if you haven't already.
2. Create a new Conda environment with the necessary libraries:

```bash
conda create -n license_plate_detection python=3.10
conda activate license_plate_detection
pip install fastapi uvicorn numpy pillow opencv-python-headless requests ultralytics easyocr python-multipart
```

### Run the API

1. Save the FastAPI code in a file named 

main.py

.
2. Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## Usage

### Endpoint: `/detect-license-plate/`

#### Method: POST

#### Description

This endpoint accepts an image file and returns the processed image with detected license plates and their bounding boxes.

#### Request

- **URL**: `http://localhost:8000/detect-license-plate/`
- **Method**: [`POST`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2Fdevelopment%2Fcomputer-vision%2Ffast-api-license-plate%2FREADME.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A39%2C%22character%22%3A13%7D%7D%5D%2C%22b89f4dd5-2277-4f4c-8f7f-8526e22ce16b%22%5D "Go to definition")
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: Form-data with a key [`file`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2Fdevelopment%2Fcomputer-vision%2Ffast-api-license-plate%2FREADME.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A22%2C%22character%22%3A30%7D%7D%5D%2C%22b89f4dd5-2277-4f4c-8f7f-8526e22ce16b%22%5D "Go to definition") containing the image file.

#### Example Request

```python
import requests

url = 'http://localhost:8000/detect-license-plate/'
files = {'file': open('./img/5686RDH.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.status_code)

# Save the received image to a file
if response.status_code == 200:
    with open('output.jpg', 'wb') as f:
        f.write(response.content)
    print("Processed image saved as 'output.jpg'")
else:
    print("Error:", response.text)  # Show the error message if status is not 200
```

### Response

- **Content-Type**: `image/jpeg`
- **Body**: The processed image with detected license plates and bounding boxes.

### Example Response

- **Status Code**: `200 OK`
- **Body**: The processed image saved as `output.jpg`.

## Notes

- Ensure the models ([`coco_model`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2Fdevelopment%2Fcomputer-vision%2Ffast-api-license-plate%2FREADME.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A100%2C%22character%22%3A22%7D%7D%5D%2C%22b89f4dd5-2277-4f4c-8f7f-8526e22ce16b%22%5D "Go to definition") and [`np_model`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2Fdevelopment%2Fcomputer-vision%2Ffast-api-license-plate%2FREADME.md%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A100%2C%22character%22%3A39%7D%7D%5D%2C%22b89f4dd5-2277-4f4c-8f7f-8526e22ce16b%22%5D "Go to definition")) are properly defined and loaded in your 

main.py

.
- Adjust the image paths and URLs as necessary for your setup.