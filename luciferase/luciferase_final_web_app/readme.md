
Introduction:

This code implements an unsupervised clustering algorithm that uses the K-means algorithm to cluster luciferase images (glowing plant images). The K-means algorithm is a popular machine learning algorithm used for unsupervised clustering. The algorithm takes as input a dataset and a user-defined value k, which represents the number of clusters. It then iteratively assigns each data point to the nearest centroid and updates the centroid location based on the newly assigned points. This process continues until the algorithm converges, i.e., the centroids no longer change significantly.

This code uses the VGG16 pre-trained model for feature extraction from luciferase images. The feature extraction process involves pre-processing an image, loading it into the VGG16 model, and using the last-but-one layer's output as features. The extracted features are then passed through a Principal Component Analysis (PCA) algorithm to reduce the dimensionality of the features from 4096 to 50. The reduced features are then used as input to the K-means clustering algorithm.

The Kmeans model is trained on 302 luciferase images and is saved as a pkl file along with the PCA pkl file. These two files are supposed to be uploaded into the streamlitApp_V2-2.ipynb before running the code. Only then the demo app will have access to required models.

Note: Below sections are for the code that is found in streamlitApp_V2-2.ipynb

Installation:

This code requires Streamlit and Pyngrok libraries, which can be installed using pip:

!pip install streamlit
!pip install pyngrok
Other required libraries include numpy, tensorflow, keras, and PIL. These libraries can be installed using pip.

Usage:

The code is contained in the app.py file, and can be run using Streamlit:

Once the Streamlit app is launched, the user can upload a luciferase image. The app then extracts features from the uploaded image, reduces its dimensionality, and uses the K-means algorithm to predict which cluster the image belongs to. The app displays the predicted cluster ID and the location of the cluster centroids.

The app also collects any new luciferase images uploaded by the user and stores them in a CSV file. Once the number of new images reaches three, the K-means algorithm retrains on the collected images and updates the cluster centroids.

Key Note:

The app was coded using Google Colab for easy and free implementation. Pyngrok was used to generate a public URL for the Streamlit app that you can run locally and share with others for testing, without having to deploy it to a remote server.

In particular, the Pyngrok package is used to create and manage the secure tunnel between your local machine and the internet, and to retrieve the public URL generated for your Streamlit app MVP. To try this app with Pyngrok, you need to run all the cells in this file to start the free app, and start a Google Colab instance. When deploying this project as an app, sophisticated options like AWS can be considered in the future.

Code Structure:

The code is divided into several functions that handle different tasks:

extract_features() - This function pre-processes an input image and extracts features from it using the VGG16 pre-trained model.
preprocess_input_img_for_kmeans() - This function calls the extract_features() function and uses PCA to reduce the dimensionality of the extracted features.
load_model() - This function loads a pre-trained K-means model from a pickle file.
save_new_model() - This function saves an updated K-means model to a pickle file.
app.py - This is the main file that runs the Streamlit app. It calls the above functions to extract features from input images, preprocess them, and predict the cluster IDs. It also handles storing new images and updating the K-means model when the number of new images reaches three.

Conclusion:

This code implements an unsupervised clustering algorithm using the K-means algorithm to cluster luciferase images. The code uses the VGG16 pre-trained model for feature extraction and PCA for feature dimensionality reduction. The app allows users to upload images and get predictions on their cluster IDs. The app also collects new images and retrains the K-means algorithm once the number of new images reaches three.


Step by step instruction to test the ngrok app:

- Start the google colab instance
- Upload the ML model pickle file and pca pickle file
- Run all the cells from the beginning, make sure to wait for the previous cell to finish its execution
- Copy the ngrok url that is generated, looks something like this: http://5ad5-35-237-28-92.ngrok-free.app
- Paste this url into any browser and test the application
