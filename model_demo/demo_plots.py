# Demo Day Plots
# Script to generate the plots for the Demo Day streamlit page.

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import v2

import itertools

# Custom DataLoader - have it in the same folder as this script 
from create_dataset import CustomFireImagesDataset

def load_dataset_path(dataset_selection,dataset_options):

    if dataset_selection == dataset_options[0]:
        
        # DeepFire Dataset
        path_dataset = '../data_preprocessing/02_forest_fire_dataset/'
        annot_file_test = 'labels_02_test_dataset_prep.csv'
        # mean and std calculated during training
        statistics = [[0.4249, 0.3509, 0.2731],[0.2766, 0.2402, 0.2612]]
        img_class = ('non-fire','fire')

    elif dataset_selection == dataset_options[1]:
        
        # WildFire Dataset
        path_dataset = '../data_preprocessing/03_the_wildfire_dataset_250x250/'
        annot_file_test = 'labels_03_test_dataset.csv'
        # mean and std calculated during training
        statistics = [[0.4158, 0.4036, 0.3758],[0.2733, 0.2565, 0.2799]]
        img_class = ('non-fire','fire')

    else:
        # Danger Dataset - cats and dogs
        path_dataset = '../data_preprocessing/danger_dataset_250x250'
        annot_file_test = 'labels_danger_dataset_demoday.csv'
        # placeholder statistics, we don't really need them for this
        statistics = [[0,0,0],[1,1,1]]
        img_class = ('cats','dogs')

    return path_dataset,annot_file_test,statistics,img_class


@st.cache_data
def load_data(path_to_dataset,annot_file_test,stats):

    full_img_path = os.path.join(path_to_dataset,annot_file_test)
    df_images = pd.read_csv(full_img_path,header = None)
    df_images.columns = ['item','label']

    # make dataset and dataloader

    curr_transf = transforms.Compose([transforms.v2.ToDtype(torch.float),
                                  transforms.Normalize([0,0,0],[255,255,255]),
                                  transforms.Resize(size=(128,128)),
                                  transforms.Normalize(stats[0],stats[1])
                                 ])
    
    dataset = CustomFireImagesDataset(annot_file_test,path_to_dataset, transform = curr_transf)

    data_loader = DataLoader(dataset,batch_size = 1, shuffle = False)
    
    return df_images, data_loader

@st.cache_resource
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

model_dict = load_all_models()

# Print Title
st.title('Image Classification of Forest Fires with Deep Neural Networks')


# Contents of the sidebar

with st.sidebar:

    # 1. Select a Dataset
    dataset_options = ('DeepFire', 'Wildfire Dataset', 'DANGER')
    dataset_selection = st.selectbox('Select a Dataset', dataset_options)

    path_dataset, annot_file_test,statistics, img_class = load_dataset_path(dataset_selection,dataset_options)
    df_images, data_loader = load_data(path_dataset,annot_file_test,statistics)


    # 2. Select an image class
    class_choice = st.radio('Select an Image Class',img_class, horizontal = True)

    # filter df_images by class 

    if class_choice == img_class[0]:
        
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

    # st.write(selected_image)

    # 4. display image
    img_caption = f'True Label: {img_class[label]}'
    st.image(img_path, channels = 'rgb', caption = img_caption, use_column_width = True)


    # 5. Click to make predictions
    click = st.button('Make predictions')

# Just to add two columns to name each model
col1, col2= st.columns(2,gap = 'large')

# make predictions if button is clicked
if click:
    with col1:
        st.header('\tVGG19')

    with col2:
        st.header('ResNet18')

    # Find image index in df_images
    selected_row = df_images[df_images['item'] == selected_image]

    image_index = selected_row['item'].index.item()

    sample = next(itertools.islice(data_loader,image_index,image_index+1))
    img,label = sample

    
    # Perform Prediction with each model
    softmax = torch.nn.Softmax(dim = 1)

    predictions = []
    

    for model in model_dict.values():
    
        pred_logits = model(img)
        pred_prob = softmax(pred_logits).squeeze()

        predictions.append(pred_prob.detach().numpy())

    hard_pred = np.argmax(predictions,axis = 1)
        

    model_name = ['VGG19','ResNet18','VGG19','ResNet18']    
    dataset_name = ['DeepFire','DeepFire','WildFire','WildFire']
    labels = ['non-fire','fire'] 

    plt.style.use("seaborn-v0_8-darkgrid")

    fig,ax_list = plt.subplots(2,2)

    for k,ax in enumerate(ax_list.ravel()):
        
        pred = predictions[k]
        ax.bar([0,1],pred)
        ax.set_ylim(0,1)
        ax.set_xticks([0,1],labels)

        ax.set_ylabel('Probability')

        curr_pred = labels[hard_pred[k]]

        curr_title = f'Prediction: {curr_pred}\nTrained on {dataset_name[k]}'

        ax.set_title(curr_title)

    plt.tight_layout()

    st.pyplot(fig)