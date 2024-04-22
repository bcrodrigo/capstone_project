# Script to deploy a streamlit app showcasing the predictions from trained models
# Note that the predictions were done in advance and saved into separate dataframes
# in order to minimize resource usage

import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

@st.cache_data
def load_data(dataset_selection,dataset_options):

    if dataset_selection == dataset_options[0]:

        # DeepFire Dataset Selected
        filename = './model_demo/prediction_files/predictions_DeepFire.csv'
        img_class = ('non-fire','fire')
        path_dataset = './model_demo/02_forest_fire_dataset_128x128/'

    elif dataset_selection == dataset_options[1]:

        # WildFire Dataset Selected
        filename = './model_demo/prediction_files/predictions_Wildfire Dataset.csv'
        img_class = ('non-fire','fire')
        path_dataset = './model_demo/03_the_wildfire_dataset_128x128/'

    elif dataset_selection == dataset_options[2]:

        # Danger Dataset - cats and dogs - Selected
        filename = './model_demo/prediction_files/predictions_DANGER.csv'
        img_class = ('cats','dogs')
        path_dataset = './model_demo/danger_dataset_128x128/'

    df = pd.read_csv(filename)

    return df,img_class,path_dataset

# Print Title
st.title('Image Classification of Forest Fires with Deep Neural Networks')

# Contents of the sidebar
with st.sidebar:

    # 1. Select a Dataset
    dataset_options = ('DeepFire', 'Wildfire Dataset', 'DANGER')
    dataset_selection = st.selectbox('Select a Dataset', dataset_options)

    df_images_pred, img_class, path_dataset = load_data(dataset_selection,dataset_options)

    # 2. Select an image class
    class_choice = st.radio('Select an Image Class',img_class, horizontal = True)

    # filter dataframe by class 
    if class_choice == img_class[0]:
        
        df_filtered = df_images_pred.query('label == 0')

    else:
        df_filtered = df_images_pred.query('label == 1')

    df_filtered.reset_index(inplace = True)

    max_ind = df_filtered.shape[0]

    # 3. Select an image from the dataset
    selected_ind = st.slider('Select an image in the dataset', 1, max_ind, 1)
    ind = selected_ind - 1

    selected_image = df_filtered.at[ind,'item']
    label = df_filtered.at[ind,'label']

    img_path = os.path.join(path_dataset,selected_image)

    # 4. display image
    img_caption = f'True Label: {img_class[label]}'
    st.image(img_path, channels = 'rgb', caption = img_caption, use_column_width = True)

    # 5. Click to make predictions
    click = st.button('Make Predictions')

# Just to add two columns to name each model
col1, col2 = st.columns(2, gap = 'large')

# make predictions if button is clicked
if click:
    with col1:
        st.header('\tVGG19')

    with col2:
        st.header('ResNet18')

    # select row from dataframe
    selected_row = df_images_pred[df_images_pred['item'] == selected_image]

    # make an array with soft predictions (4 models x 2 classes)
    predictions = selected_row.iloc[:,2:].values.reshape(4,2)

    # calculate hard predictions
    hard_pred = np.argmax(predictions, axis = 1)

    model_name = ['VGG19','ResNet18','VGG19','ResNet18']    
    dataset_name = ['DeepFire','DeepFire','WildFire','WildFire']
    labels = ['non-fire','fire'] 

    plt.style.use("seaborn-v0_8-darkgrid")

    fig,ax_list = plt.subplots(2,2)

    for k,ax in enumerate(ax_list.ravel()):
        
        pred = predictions[k,:]
        ax.bar([0,1],pred)
        ax.set_ylim(0,1)
        ax.set_xticks([0,1],labels)

        ax.set_ylabel('Probability')

        curr_pred = labels[hard_pred[k]]

        curr_title = f'Prediction: {curr_pred}\nTrained on {dataset_name[k]}'

        ax.set_title(curr_title)

    plt.tight_layout()

    st.pyplot(fig)