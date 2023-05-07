
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
from sklearn.cluster import KMeans
from io import BytesIO
from flask_cors import CORS

np.random.seed(52)
tf.keras.utils.set_random_seed(52)

app = Flask(__name__)

CORS(app)

# Load the machine learning model
with open('../Models/kmodel.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../Models/pca.pkl', 'rb') as f:
    pca_model = pickle.load(f)
    
with open('../Models/threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)
    
with open('../Models/distances.pkl', 'rb') as f:
    distances = pickle.load(f)

with open('../Models/memory_buffer.pkl', 'rb') as f:
    memory_buffer1 = pickle.load(f)

# Define an endpoint for the model
@app.route('/predict', methods=['POST'])
def predict():
    
    with open('../Models/kmodel.pkl', 'rb') as f:
        model = pickle.load(f)
        
    with open('../Models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
        
    with open('../Models/distances.pkl', 'rb') as f:
        distances = pickle.load(f)

    with open('../Models/memory_buffer.pkl', 'rb') as f:
        memory_buffer1 = pickle.load(f)
        
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
    print([x1,y1])
    cluster_id_prediction = model.predict(
        np.array([x1, y1]).reshape((1, -1))
    )[0]
    print(cluster_id_prediction)
    cluster_distance = min(   
                    preprocessing.normalize(
                        model.transform(np.array([x1, y1]).reshape((1, -1)))
                    )[0]
                )
    print(cluster_distance)
    dist = distances[0, cluster_id_prediction]
    # Return the prediction
    return {'cluster': int(cluster_id_prediction),'img_dist':dist,'threshold':threshold}


def process_image(im):
    try:
        
        with open('../Models/pca.pkl', 'rb') as f:
            pca_model = pickle.load(f)
        
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
        return vis_data[0][0], vis_data[0][1]
    except Exception as ex:
        # skip all exceptions for now
        print(ex)
        pass


# Function to update centroids incrementally
def incremental_update_centroids(kmeans, new_data, learning_rate=0.1):
    for data_point in new_data:
        data_point = np.array(data_point).reshape(1, -1)  # Ensure the data point has the correct dimensions
        nearest_centroid_idx = np.argmin(np.sum((kmeans.cluster_centers_ - data_point)**2, axis=1))
        kmeans.cluster_centers_[nearest_centroid_idx] += learning_rate * (np.squeeze(data_point) - kmeans.cluster_centers_[nearest_centroid_idx])

# Function to retrain K-Means on combined data
def retrain_kmeans(kmeans, memory_buffer, new_data):
    combined_data = np.vstack([memory_buffer, new_data])
    kmeans = KMeans(n_clusters=kmeans.n_clusters, init=kmeans.cluster_centers_, n_init=1, random_state=42)
    kmeans.fit(combined_data)
    return kmeans

@app.route('/retrain', methods=['POST'])
def retrain():
    with open('../Models/kmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../Models/pca.pkl', 'rb') as f:
        pca_model = pickle.load(f)
        
    with open('../Models/threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)
        
    with open('../Models/distances.pkl', 'rb') as f:
        distances = pickle.load(f)

    with open('../Models/memory_buffer.pkl', 'rb') as f:
        memory_buffer1 = pickle.load(f)
        
    # Read the images parameter from the request body
    images_b64 = request.json['images']
    
    # Decode the base64-encoded images to bytes
    image_bytes_list = [base64.b64decode(image_b64) for image_b64 in images_b64]
    
    # Convert the bytes to images
    images = [Image.open(io.BytesIO(image_bytes)) for image_bytes in image_bytes_list]
    
    # Resize the images to a target size
    target_size = (224, 224)
    images_resized = [img.resize(target_size) for img in images]
    kmeans = model
    # Preprocess the input data
    x_list, y_list = [], []
    for img in images_resized:
        x, y = process_image(img)
        x_list.append(x)
        y_list.append(y)
    combined_data = np.column_stack((x_list, y_list))
    # Update centroids incrementally
    incremental_update_centroids(kmeans,combined_data)

    # Retrain K-Means periodically
    memory_buffer = memory_buffer1
    retraining_interval = 10  # Retrain K-Means every 10 new data points

    for i in range(0, len(combined_data), retraining_interval):
        batch_new_data = combined_data[i:i + retraining_interval]
        kmeans = retrain_kmeans(kmeans, memory_buffer, batch_new_data)
        memory_buffer = np.vstack([memory_buffer, batch_new_data])  # Update the memory buffer with new data
    # get the distances between data points and their assigned centroids
    distances = kmeans.transform(memory_buffer)

    # calculate the median absolute deviation (MAD) of the distances
    med = np.median(distances)
    abs_dev = np.abs(distances - med)
    mad = np.median(abs_dev)

    # calculate the threshold as a multiple of the MAD
    threshold = 3 * mad
    pickle.dump(kmeans, open('../Models/kmeans.pkl', 'wb'))
    pickle.dump(memory_buffer, open('../Models/memory_buffer.pkl', 'wb'))
    pickle.dump(threshold, open('../Models/threshold.pkl', 'wb'))
    pickle.dump(distances, open('../Models/distances.pkl', 'wb'))
    return {'status':"ok"}

if __name__ == '__main__':
     app.run(host='ec2-13-59-206-36.us-east-2.compute.amazonaws.com', port=8002)
