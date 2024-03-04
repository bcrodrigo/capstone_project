import numpy as np
import pandas as pd
import os

def annotations_file():
    '''
    Function that creates the annotations file (.csv) for an image dataset.
    The dataset is assumed to be divided into two folders according to the labels.

    '''

    path_to_dataset = '../data/'
    dataset_foldername = '02_forest_fire_dataset'
    fire_images_folder = 'training/fire'
    nonfire_images_folder = 'training/nofire'

    curr_path = path_to_dataset + dataset_foldername + '/' + fire_images_folder
    curr_path_2 = path_to_dataset + dataset_foldername + '/' + nonfire_images_folder

    fire_img_list = os.listdir(curr_path)
    nonfire_img_list = os.listdir(curr_path_2)

    fire_labels = np.ones(len(fire_img_list),dtype = int)
    nonfire_labels = np.zeros(len(nonfire_img_list),dtype = int)

    fire_df = pd.DataFrame(data = fire_img_list, columns=['item'])
    fire_df['label'] = fire_labels
    fire_df['item'] = fire_df['item'].apply(lambda x: fire_images_folder + '/' + x)

    nonfire_df = pd.DataFrame(data = nonfire_img_list, columns = ['item'])
    nonfire_df['label'] = nonfire_labels
    nonfire_df['item'] = nonfire_df['item'].apply(lambda x: nonfire_images_folder + '/' + x )

    all_images_df = pd.concat([fire_df,nonfire_df],axis = 0)
    all_images_df.info()

    output_filename = 'labels_' + dataset_foldername + '.csv'

    # save annotations file as csv without header and index
    all_images_df.to_csv(path_to_dataset + dataset_foldername + '/' + output_filename, index = False, header = False)



def image_sizes(annotations_file,path_to_file):
    '''
    Placeholder. I want this function to extract the size of each image and get a sense of the distribution.
    '''
    full_path = path_to_file + '/' + annotations_file

    annotations = pd.read_csv(full_path)


