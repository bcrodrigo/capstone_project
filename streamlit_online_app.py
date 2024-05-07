# Script to test deploying a streamlit app
# minimizing resource usage
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

# to set the page to wide layout by default
def wide_space_default():
    st.set_page_config(layout = 'wide')

wide_space_default()

@st.cache_data
def load_data(dataset_selection,dataset_options):

    if dataset_selection == dataset_options[0]:

        # DeepFire Dataset
        filename = './model_demo/prediction_files/predictions_DeepFire.csv'
        img_class = ('non-fire','fire')
        path_dataset = './model_demo/02_forest_fire_dataset_128x128/'

    elif dataset_selection == dataset_options[1]:

        # WildFire Dataset
        filename = './model_demo/prediction_files/predictions_WildFire_Dataset.csv'
        img_class = ('non-fire','fire')
        path_dataset = './model_demo/03_the_wildfire_dataset_250x250/'

    elif dataset_selection == dataset_options[2]:

        # Danger Dataset - cats and dogs
        filename = './model_demo/prediction_files/predictions_DANGER.csv'
        img_class = ('cats','dogs')
        path_dataset = './model_demo/danger_dataset_128x128/'

    df = pd.read_csv(filename)

    return df,img_class,path_dataset

# Print Title
st.title('Image Classification of Forest Fires with Deep Neural Networks')

expander = st.expander("Instructions")

instructions = '''
1. Select a Dataset from the sidebar dropdown
2. Select an Image Class
3. Choose an Image in the Dataset
4. Click `Make Predictions`
'''
expander.markdown(instructions)

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

if dataset_selection == dataset_options[2]:
    disclaimer = '''
    ### WARNING

    The models were only trained to classify fire and non-fire images.
    
    However, if we input an appropriately-sized RGB image, the models will still perform a prediction, even if it doesn't make sense!
    '''
    st.markdown(disclaimer)

# make predictions if button is clicked
if click:

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

    # testing plotting on different containers
    row1 = st.columns(2)
    row2 = st.columns(2)

    for ind,col in enumerate(row1 + row2):
        tile = col.container(border = True)

        curr_subheader = f'{model_name[ind]} trained on {dataset_name[ind]}'
        tile.subheader(curr_subheader)

        with tile:

            fig = plt.figure()

            pred = predictions[ind,:]
            plt.bar([0,1],pred)
            plt.ylim(0,1)
            plt.yticks(np.arange(0,1.1,0.2), fontsize = 15)
            plt.xticks([0,1],labels, fontsize = 15)
            plt.ylabel('Probability', fontsize = 20)

            curr_pred = labels[hard_pred[ind]]

            # if hard prediction is fire, colour text red
            if hard_pred[ind]:
                color = 'red'
            else:
                color = 'grey'

            curr_title = f'Prediction: :{color}[{curr_pred}]'
            st.write(curr_title)

            st.pyplot(fig)

    github_link = 'Back to [GitHub repository](https://github.com/bcrodrigo/capstone_project)'
    st.markdown(github_link, unsafe_allow_html = True)