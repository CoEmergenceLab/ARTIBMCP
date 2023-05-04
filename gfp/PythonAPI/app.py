import requests
import numpy as np
import base64
from PIL import Image
from tensorflow.keras.preprocessing import image
from io import BytesIO

# Endpoint for the machine learning model
model_endpoint = 'http://127.0.0.1:8002/predict'

# Open the image file and read its contents
with open('test.jpg', 'rb') as f:
    image_bytes = f.read()

# Encode the image as base64
image_b64 = base64.b64encode(image_bytes).decode('utf-8')

# Define the request payload
payload = {'image': image_b64}



# print(str(img))
# # Send the image data to the machine learning model
# headers = {'content-type': 'image/jpeg'}
# response = requests.post(model_endpoint, data=str(img), headers=headers)

# Get the prediction from the machine learning model
# prediction = response.json()


# Make the request
response = requests.post(model_endpoint, json=payload)

# Print the response
print('\nCluster ID :\n')
print(response.json())