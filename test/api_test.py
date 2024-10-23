import requests

url = 'http://172.20.10.2:8000/detect-license-plate/'
files = {'file': open('../img/4472SKF.jpg', 'rb')}
response = requests.post(url, files=files)

# Comprobar si la respuesta es JSON válida y mostrarla
try:
    response_data = response.json()
    print("Respuesta en JSON:")
    print(response_data)
except ValueError:
    # Si no es JSON, mostrar el contenido de texto
    print("Respuesta de texto:")
    print(response.text)
    
    
# Guarda la imagen recibida en un archivo
if response.status_code == 200:
    with open('output.jpg', 'wb') as f:
        f.write(response.content)
    print("Imagen procesada guardada como 'output.jpg'")
else:
    print("Error:", response)  # Mostrar el mensaje de error en caso de que no sea 200
