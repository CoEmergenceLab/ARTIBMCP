
import io
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from keras import applications
from sklearn import preprocessing
import numpy as np
import pickle
import base64
from PIL import Image

from io import BytesIO
from flask_cors import CORS



app = Flask(__name__)

CORS(app)

# Load the machine learning model
with open('gfp.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca_model = pickle.load(f)



# Define an endpoint for the model
@app.route('/predict', methods=['POST'])
def predict():
   # Read the image parameter from the request body
    image_b64 = request.json['image']
    
    # Decode the base64-encoded image to bytes
    image_bytes = base64.b64decode(image_b64)
    
    # Convert the bytes to an image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image to a target size
    target_size = (224, 224)
    img = img.resize(target_size)
    
    # Preprocess the input data
    x1, y1 = process_image(img)
    # Make a prediction using the modelx1, y1 = process_test_image(target, pca_model)
    cluster_id_prediction = model.predict(
        np.array([x1, y1]).reshape((1, -1))
    )[0]
    print(cluster_id_prediction)

    # Return the prediction
    return {'cluster': int(cluster_id_prediction)}


def process_image(im):
    try:
        # newsize = (224, 224)
        # img = im.resize(newsize)
        # # convert image to numpy array
        x = image.img_to_array(im)
        # the image is now in an array of shape (3, 224, 224)
        # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        model_k = applications.vgg16.VGG16(
            weights="imagenet", include_top=False, pooling="avg"
        )

        # extract the features
        features = model_k.predict(x)[0]
        # convert from Numpy to a list of values
        features_arr = np.char.mod("%f", features)
        feature_list = ",".join(features_arr)
        transformed = feature_list.split(",")

        # convert image data to float64 matrix. float64 is need for bh_sne
        x_data = np.asarray(transformed).astype("float64")
        x_data = x_data.reshape((1, -1))
        # perform t-SNE

        vis_data = pca_model.transform(x_data)

        # convert the results into a list of dict
        results = []
        print(vis_data)
        return vis_data[0][0], vis_data[0][1]
    except Exception as ex:
        # skip all exceptions for now
        print(ex)
        pass

if __name__ == '__main__':
    app.run(port=8002)




