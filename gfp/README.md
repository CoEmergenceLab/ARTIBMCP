# GFP Image Cluster Prediction and Continual Learning Angular Application

> Aim - In this project we intend to establish meaningful dialogue between plant and human beings, that is contextualized to how plants and humans will have to adapt to in order to survive. The central objective is to study plant images, and based on its results produce meaningful response that helps plant to grow. The response involves changes in the environmental stresses that includes temperature,sound, light, parasites, and so on. In the initial phase, we will be concentrating on Sound parameter.

  

Our dataset includes 3 types of images

- Bioluminescence

- Fluorescence

- Thermal

In this notebook, we will examine Green fluorescent protein(GFP) which is one of the Fluorescence plants (glows in dark).

  

## Introduction

This technical documentation is intended to provide a comprehensive overview of the Image Cluster Prediction and Continual Learning Angular Application. The application allows users to upload images and receive cluster predictions based on a machine learning (ML) model. It also enables users to trigger a continual learning pipeline for their ML model by uploading a set of images for the model to learn. The Angular app calls a Python API, which in turn runs the ML model to obtain predictions. The application is hosted on Azure DevOps.

  

## Development

We have developed a machine learning model to cluster an incoming plant image based on its features. Based on the resultant cluster, sound is mapped to the respective image and helps in its growth.

  

Execution of the model notebook has following requirements:

- Tensorflow version - 2.9.2

- CPU Memory - >= 4GB

- data.csv file - A csv file that consists the list of all images filename for the easement in the data analysis. This csv file is uploaded in **/dataset/** directory

  

## Application Architecture

The application consists of three main components:

  

-  **Angular Frontend:** Provides a user interface for uploading images, displays predictions, and triggers the continual learning pipeline.

- **Python API:** Acts as a bridge between the Angular frontend and the ML model, handling requests and responses.

- **ML Model:** Processes the image data, provides cluster predictions, and incorporates new data through the continual learning pipeline.

> *Angular Frontend*

The Angular frontend is developed using Angular, a popular web application framework. The frontend provides a user-friendly interface for interacting with the application's features. Key components include:

- Image Upload: Users can upload an image to receive a cluster prediction. Uploaded images are validated for file type and size before being sent to the Python API.

- Prediction Display: Shows the cluster prediction, the distance of the image from the cluster centroid, and whether the image falls within the defined threshold.

- Continual Learning: Allows users to upload a set of images for the ML model to learn from, improving its prediction accuracy over time.

>Python API

The Python API serves as an intermediary between the Angular frontend and the ML model. It handles HTTP requests and responses, ensuring that data is properly transmitted between the frontend and the model. Key components include:

- Flask: A lightweight web framework used to create the API endpoints.

- Image Processing: Converts the uploaded image into a format that can be processed by the ML model.

- Prediction Handling: Sends processed image data to the ML model, receives predictions, and returns them to the frontend.

- Continual Learning: Receives sets of images from the frontend and triggers the continual learning pipeline in the ML model.

>ML Model

The ML model is responsible for processing image data, providing cluster predictions, and incorporating new data through the continual learning pipeline. The model utilizes a clustering algorithm (e.g., K-means) to group images based on their features. The continual learning pipeline helps the model adapt to new data and improve its prediction accuracy.


## AWS Deployment

The Angular app and the Python API are hosted on AWS, a cloud-based platform that facilitates continuous integration and deployment.

- Angular UI – [http://ec2-3-15-219-18.us-east-2.compute.amazonaws.com:4200/](http://ec2-3-15-219-18.us-east-2.compute.amazonaws.com:4200/)

- Python API – [http://ec2-3-145-23-188.us-east2.compute.amazonaws.com:8002/](http://ec2-3-145-23-188.us-east-2.compute.amazonaws.com:8002/)

## Usage

To use the Image Cluster Prediction and Continual Learning Angular Application, follow these steps:

- Access the application through the provided URL.

- Upload an image using the "Upload Image" button.

- View the cluster prediction, distance from the centroid, and threshold information.

- To trigger the continual learning pipeline, upload a set of images using the "Upload Images for Learning" button.

## Conclusion
The Image Cluster Prediction and Continual Learning Angular Application is a powerful tool for obtaining cluster predictions based on an ML model and enhancing the model's accuracy through continual learning. With its user-friendly interface, robust API, and seamless integration with Azure DevOps, this application streamlines the process of obtaining predictions and updating the ML model with new data. By leveraging the power of clustering algorithms and continual learning techniques, users can derive valuable insights from their image data and improve their ML model's performance over time.