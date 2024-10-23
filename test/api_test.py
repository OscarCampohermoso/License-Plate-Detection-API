import requests

url = 'http://localhost:8000/detect-license-plate/'
files = {'file': open('./img/5686RDH.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.status_code)

# Guarda la imagen recibida en un archivo
if response.status_code == 200:
    with open('output.jpg', 'wb') as f:
        f.write(response.content)
    print("Imagen procesada guardada como 'output.jpg'")
else:
    print("Error:", response.text)  # Mostrar el mensaje de error en caso de que no sea 200
