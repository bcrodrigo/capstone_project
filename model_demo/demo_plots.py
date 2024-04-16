# Demo Day Plots
# Notebook to generate the plots for the Demo Day streamlit page.

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import torch

from torchvision.io import read_image
from torchvision import datasets

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2

import itertools

def load_dataset_path(dataset_selection,dataset_options):

    if dataset_selection == dataset_options[0]:

        path_dataset = '../data_preprocessing/02_forest_fire_dataset/'
        annot_file_test = 'labels_02_test_dataset_prep.csv'

    elif dataset_selection == dataset_options[1]:

        path_dataset = '../data_preprocessing/03_the_wildfire_dataset_250x250/'
        annot_file_test = 'labels_03_test_dataset.csv'

    else:
        pass
        print('TO DO create a DANGER dataset')

    return path_dataset,annot_file_test


@st.cache_data
def load_data(path_03_dataset,annot_file_test):
    full_img_path = os.path.join(path_03_dataset,annot_file_test)
    df_images = pd.read_csv(full_img_path,header = None)
    df_images.columns = ['item','label']

    # make dataset and dataloader

    curr_transf = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(size=(128,128))
        ])

    dataset = datasets.ImageFolder(path_03_dataset,transform = curr_transf)
    
    data_loader = DataLoader(dataset,batch_size = 1, shuffle = False)
    
    return df_images, data_loader

@st.cache_data
def load_all_models():
    """To load all trained models to do predictions. Note all models are set
    automatically to `eval()` mode.
    
    Returns
    -------
    dictionary
        All loaded models in a dictionary.
    """
    model_dict = {'vgg_02':torch.load('../model_demo/vgg19_02dataset.pth'),
                  'resnet_02':torch.load('../model_demo/resnet18_02dataset.pth'),
                  'vgg_03':torch.load('../model_demo/vgg19_03dataset.pth'),
                  'resnet_03':torch.load('../model_demo/resnet18_03dataset.pth')
                 }
    
    model_dict['vgg_02'].eval()
    model_dict['resnet_02'].eval()
    model_dict['vgg_03'].eval()
    model_dict['resnet_03'].eval()
    
    return model_dict


img_class = ('non-fire','fire')

# Print Title
st.title('Image Classification of Forest Fires with Deep Neural Networks')


# 1. Select a Dataset
dataset_options = ('DeepFire', 'Wildfire Dataset', 'DANGER')
dataset_selection = st.selectbox('Select a Dataset', dataset_options)

path_dataset, annot_file_test = load_dataset_path(dataset_selection,dataset_options)
df_images, data_loader = load_data(path_dataset,annot_file_test)


# 2. Select an image class
class_choice = st.radio('Select an Image Class',img_class, horizontal = True)

# filter df_images by class 

if class_choice == 'non-fire':
    
    df_filtered = df_images.query('label == 0')

else:
    df_filtered = df_images.query('label == 1')

df_filtered.reset_index(inplace = True)

max_ind = df_filtered.shape[0] - 1


# 3. Select an image from the dataset
ind = st.slider('Select an image in the dataset', 0, max_ind, 0)

selected_image = df_filtered.at[ind,'item']
label = df_filtered.at[ind,'label']

img_path = os.path.join(path_dataset,selected_image)


# 4. display image
img_caption = f'True Label: {img_class[label]}'
st.image(img_path, channels = 'rgb', caption = img_caption)


# 5. make predictions
click = st.button('Make predictions')

# make predictions if button is clicked
if click:

    # Find image index in df_images
    selected_row = df_images[df_images['item'] == selected_image]

    image_index = selected_row['item'].index.item()

    sample = next(itertools.islice(data_loader,image_index,image_index+1))
    img,label = sample

    
    # Perform Prediction
    model_dict = load_all_models()
    softmax = torch.nn.Softmax(dim = 1)

    predictions = []

    for model in model_dict.values():
    
        pred_logits = model(img)
        pred_prob = softmax(pred_logits).squeeze()

        predictions.append(pred_prob.detach().numpy())
        

    # display multiple plots
    model_name = ['VGG19','ResNet18','VGG19','ResNet18']

    fig,ax_list = plt.subplots(2,2)

    for k,ax in enumerate(ax_list.ravel()):
        mod = model_name[k]
        pred = predictions[k]
        ax.bar([0,1],pred)
        ax.set_ylim(0,1)
        ax.set_xticks([0,1],['non-fire','fire'])

        ax.set_ylabel('Probability')
        ax.set_title(mod)

    plt.tight_layout()

    st.pyplot(fig)
