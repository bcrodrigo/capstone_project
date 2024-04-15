This directory contains all the Jupyter Notebooks I used in this project for EDA, preprocessing, training, and model evaluation. The notebooks are numbered according to the project progression, and I've tried to name them with self-explanatory titles. Here, I'll provide some additional details for context

### 01_EDA_preprocessing
These contain the EDA and preprocessing steps for each of the original datasets I've downloaded. Some of these steps include correcting for wrong number of channels, resizing and cropping images. Lastly, the annotations files (containing image name and labels) are prepared.
### 02_Dataloader_02dataset
To test defining a custom `DataLoader` class for the [DeepFire Dataset](https://doi.org/10.1155/2022/5358359).
### 02.5_Checking_Datatypes
Notebook to double check on datatypes and prevent issues for arising during training.
### 03_Implementing_LeNet5
Here I implement LeNet5 and train it with two datasets
- CIFAR10
- DeepFire dataset
Note that this notebook is only for learning and troubleshooting purposes.
### 03_VGG19/Resnet18_TransferLearning
In these notebooks I perform Transfer Learning on VGG19 and ResNet18: all layers are frozen and the last layer is replaced with a fully-connected layer that outputs 2 classes: `non-fire` and `fire`.
The weights of the pre-trained models are downloaded from `torchvision.models` and the results of the training are further analyzed.
### 03.5_Model_Evaluation
I compare VGG19 and ResNet18 against each other and against the results shown in the [DeepFire Dataset article](https://doi.org/10.1155/2022/5358359). Some evaluation metrics include: confusion matrix, accuracy, precision, recall, F1 score, AUC score and ROC curve.
### 04_VGG19/ResNet18_TransferLearning_03dataset
In these notebooks I perform Transfer Learning to re-train VGG19 and ResNet18, this time on the [Wildfire Dataset](https://doi.org/10.3390/f14091697).
