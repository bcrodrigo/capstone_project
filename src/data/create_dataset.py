import numpy as np
import pandas as pd
import os

def annotations_file(path_to_dataset,dataset_foldername,img_folders):
    '''Function that creates the annotations file (.csv) for an image dataset.
    The dataset is assumed to be divided into two folders according to the labels.

    Parameters
    ----------
    path_to_dataset : str
        String with the path pointing to the folder containing the dataset
    dataset_foldername : str
        Name of the folder containing all images
    img_folders : list
        List with the folder names containing the images. Both folders are assumed to be inside 
        `dataset_foldername` and each position in the list is going to be mapped to a binary category
        img_folders[0] --> category 0
        img_folders[1] --> category 1

    Returns
    -------
    dataframe
        A dataframe contaning the list of images and their binary label

    '''
    assert isinstance(path_to_dataset,str),'path_to_dataset must be a string'
    assert isinstance(dataset_foldername,str),'dataset_foldername must be a string'
    assert isinstance(img_folders,list),'img_folders must be a list'

    Nfolders = len(img_folders)

    df_list = []

    for index in range(Nfolders):

        curr_folder = img_folders[index]

        assert isinstance(curr_folder,str),'element of img_folders must be a string'

        curr_path = path_to_dataset + dataset_foldername + '/' + curr_folder

        image_list = os.listdir(curr_path)
        labels = index * np.ones(len(image_list), dtype = int)

        temp_df = pd.DataFrame(columns = ['item'], data = image_list)

        # update each item to include the image folder 
        temp_df['item'] = temp_df['item'].apply(lambda x: curr_folder + '/' + x)

        # add column with labels
        temp_df['label'] = labels

        df_list.append(temp_df.copy())

    # concatenate dataframe list into a single file
    all_images_df = pd.concat(df_list, axis = 0, ignore_index = True)
    print(all_images_df.info())

    output_filename = 'labels_' + dataset_foldername + '.csv'

    # save annotations file as csv without header and index
    all_images_df.to_csv(path_to_dataset + dataset_foldername + '/' + output_filename, index = False, header = False)

    return all_images_df


def image_sizes(annotations_file,path_to_file):
    '''
    Placeholder. I want this function to extract the size of each image and get a sense of the distribution.
    '''
    full_path = path_to_file + '/' + annotations_file

    annotations = pd.read_csv(full_path)


