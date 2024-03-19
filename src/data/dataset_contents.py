# Functions to perform EDA on a given image dataset
import pandas as pd
import os
from glob import glob
from torchvision.io import read_image

def all_subdir_list(path_to_dataset,levels):
    """Function that makes a list of subdirectories in a dataset folder.
    
    Parameters
    ----------
    path_to_dataset : string
        Path (absolute or relative) to contents of image dataset folder. 
        The dataset is assumed to have the following structure
        dataset/
            folder1/
                subfolder1.1/
            folder2/
                subfolder2.1/
                subfolder2.2/
            folder3/

    levels : integer
        Number of nested levels in the image dataset
    
    Returns
    -------
    List
        Subdirectory list
    """
    assert isinstance(path_to_dataset,str),'path_to_dataset input parameter must be a string'
    assert isinstance(levels,int),'levels input parameter must be an integer'

    # change to directory containing images
    original_dir = os.getcwd()
    os.chdir(path_to_dataset)

    # note '/*' repeats as many times as indicated by `levels`
    search_path = '.{}'.format('/*'*levels)

    all_subdir = glob(search_path, recursive = True)

    print('Made a list with {} directories'.format(len(all_subdir)))

    os.chdir(original_dir)
    return all_subdir


def all_images_list(path_to_dataset,directory_list,label_list):
    """Function that lists all images contained in the subdirectories of a dataset,
    opens each one by one, and returns a dataframe containing all image names as well
    as their labels and size.
    
    Parameters
    ----------
    path_to_dataset : string
        Path (absolute or relative) to contents of image dataset folder. 
        The dataset is assumed to have the following structure
        dataset/
            folder1/
                subfolder1.1/
            folder2/
                subfolder2.1/
                subfolder2.2/
            folder3/

    directory_list : list
        List with all the subdirectories contained in the dataset.
    label_list : list
        List with the numeric categories for each of the directories in `directory_list`
    
    Returns
    -------
    Dataframe
        All the contents of the dataset into a dataframe containing 
        item name, label, channels, width, height
    """
    assert isinstance(path_to_dataset,str),'path_to_dataset input parameter must be a string'
    assert len(directory_list) == len(label_list),'directory_list and label_list are not the same lenght'

    # change to directory containing images
    original_dir = os.getcwd()
    os.chdir(path_to_dataset)

    # preallocate output dataframe
    output_df = pd.DataFrame()

    for label,subdir in zip(label_list,directory_list):

        temp_df = pd.DataFrame(columns = ['label','item'])
        
        # list all the contents of subdir
        temp_df['item'] = os.listdir(subdir)
        # append the path to the image name on each row of the df
        temp_df['item'] = temp_df['item'].apply(lambda x: os.path.join(subdir,x))

        # update labels
        temp_df['label'] = label
        
        output_df = pd.concat([output_df,temp_df], axis = 0)

    output_df.reset_index(drop = True, inplace = True)

    print('Completed list of images')

    output_df['channels'] = 0
    output_df['width'] = 0
    output_df['height'] = 0

    output_df['issues'] = 'no'

    print('Reading from image list')

    for index,image in enumerate(output_df['item']):

        try:
            img_shape = read_image(image).shape

            output_df.at[index,'channels'] = img_shape[0]
            output_df.at[index,'height'] = img_shape[1]
            output_df.at[index,'width'] = img_shape[2]

        except:
            output_df.at[index,'issues'] = 'yes'
            print(f'issue encountered with\nIndex {index}\nImage {image}')

            continue

    print('Finished reviewing all images')

    os.chdir(original_dir)
    return output_df