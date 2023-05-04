# POST-Natural Prototypes
> Aim - In this project we intend to establish meaningful dialogue between plant and human beings, that is contextualized to how plants and humans will have to adapt to in order to survive. The central objective is to study plant images, and based on its results produce meaningful response that helps plant to grow. The response involves changes in the environmental stresses that includes temperature,sound, light, parasites, and so on. In the initial phase, we will be concentrating on Sound parameter.  

 Our dataset includes 3 types of images 
 - Bioluminescence
 - Fluorescence
 - Thermal
 
In this notebook, we will examine Green fluorescent protein(GFP) which is one of the Fluorescence plants (glows in dark). 

## Dataset
GFP dataset is consist of 186 seedlings images. These are continuous images (time-lapse) which shows the growth of the plants gradually. 
GFP consist of floroscent plant hormone called Auxin, and as the plant grows, the hormone present in the plant becomes brighter. 
The dataset can be accessed from **/dataset/gfp** directory.

## Development
We have developed a machine learning model to cluster an incoming plant image based on its features. Based on the resultant cluster, sound is mapped to the respective image and helps in its growth.

Execution of the model notebook has following requirements:
- Tensorflow version - 2.9.2
- CPU Memory -  >= 4GB
- data.csv file - A csv file that consists the list of all images filename for the easement in the data analysis. This csv file is uploaded in **/dataset/** directory
