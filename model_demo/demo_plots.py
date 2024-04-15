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

    # # define transformations
    # curr_transf = transforms.Compose([transforms.v2.ToDtype(torch.float),
    #                 transforms.Normalize([0,0,0],[255,255,255]),
    #                 transforms.Resize(size=(128,128)),
    #                              ])

    curr_transf = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(size=(128,128))
        ])

    dataset = datasets.ImageFolder(path_03_dataset,transform = curr_transf)
    
    data_loader = DataLoader(dataset,batch_size = 1, shuffle = False)
    
    return df_images, data_loader

img_class = ('non-fire','fire')

# Print Title
st.title('Image Classification of Forest Fires with Deep Neural Networks')


# 1. Select a Dataset
dataset_options = ('DeepFire', 'Wildfire Dataset', 'DANGER')
dataset_selection = st.selectbox('Select a Dataset', dataset_options)

# st.write('You selected:', dataset_selection)
path_dataset, annot_file_test = load_dataset_path(dataset_selection,dataset_options)
df_images, data_loader = load_data(path_dataset,annot_file_test)

# st.write('df_images shape[0]',df_images.shape[0])
# st.write('length of data_loader',len(data_loader))

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
st.write(click)

# make predictions if button is clicked
if click:

    # Find image index in df_images
    selected_row = df_images[df_images['item'] == selected_image]

    image_index = selected_row['item'].index.item()

    st.write('selected image index',image_index)
    st.write('selected image name',selected_row['item'].values)

    # create dataset and dataloader
    st.write('len of data loader',len(data_loader))

    sample = next(itertools.islice(data_loader,image_index,image_index+1))
    img,label = sample

    # st.image(img.squeeze(),)
