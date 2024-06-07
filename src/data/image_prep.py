# functions that perform image preprocessing
import os
import PIL.Image

from torchvision.transforms.functional import crop
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.transforms import v2

import shutil

def update_channels(path_to_dataset,df_wrong_channels):
    """Function that updates the number of channels present in an image by converting it to RGB.
    It accepts JPEG or PNG images.
    
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
    df_wrong_channels : pandas dataframe
        A dataframe created by `all_images_list()` listing all the images with 'wrong' number 
        of channels (ie != 3).
        It is assumed to have the following columns: 
        `item`, `label`,`channels`,`height`,`width`

    Returns
    -------
    Dataframe
        Original `df_wrong_channels` with an additonal column (`rgb_item`) containing 
        the names of the updated files
    """

    assert isinstance(path_to_dataset,str),'path_to_dataset input parameter must be a string'

    # change to directory containing images
    original_dir = os.getcwd()
    os.chdir(path_to_dataset)

    # add additional column
    df_wrong_channels['rgb_item'] = ''

    for i in df_wrong_channels.index:
           
        image = df_wrong_channels.at[i,'item']

        print(f'Converting {image} to RGB')

        original_image = PIL.Image.open(image)
        rgb_image = original_image.convert('RGB')

        extension = image[-4:]

        # get only the path + name of the image, without extension
        image_name = image[0:-4]

        # rename image, including path
        new_image_name = image_name + '_rgb' + extension

        # update dataframe
        df_wrong_channels.at[i,'rgb_item'] = new_image_name

        rgb_image.save(new_image_name,extension[1:])
        

    # change back to current directory
    os.chdir(original_dir)
    return df_wrong_channels


def crop_image(path_to_dataset,df_oversized,new_height,new_width):
    """Function that crops a set of images and saves a new file accordingly
    
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

    df_oversized : dataframe
        A dataframe listing all the 'oversized' images, to be cropped.
        It is assumed to have the following columns: `item`, `label`,`channels`,`height`,`width`

    new_height : integer
        Cropped image height in pixels

    new_width : integer
        Cropped image width in pixels
    
    Returns
    -------
    Dataframe
        Original `df_oversized` with three additonal columns
        - `cropped_item`: name of the updated file 
        - `new_height`: new height in pixels
        - `new_width`: new width in pixels
    """
    assert isinstance(path_to_dataset,str),'path_to_dataset should be a string'
    assert isinstance(new_height,int) and new_height > 0,'new_height should be a positive integer'
    assert isinstance(new_width,int) and new_width > 0,'new_width should be a postive integer'

    df_oversized['cropped_item'] = ''
    df_oversized['new_height'] = 0
    df_oversized['new_width'] = 0

    original_dir = os.getcwd()

    os.chdir(path_to_dataset)

    for i in df_oversized.index:

        image_path_name = df_oversized.at[i,'item']
    
        extension = image_path_name[-4:]
        image_name = image_path_name[0:-4]
    
        # rename image, including path
        cropped_image_path_name = f'{image_name}_cropped{extension}'

        # update dataframe
        df_oversized.at[i,'cropped_item'] = cropped_image_path_name
        
        # read image
        img = read_image(image_path_name)
        
        # crop image
        temp = crop(img,0,0,new_height,new_width)
        
        # save image -- need to normalize to the 0 to 1 interval
        save_image(temp/255,cropped_image_path_name)
    
        print(f'Cropped {image_path_name} to {new_height} x {new_width}')


    os.chdir(original_dir)
    return df_oversized

def resize_all_images(path_to_dataset,dir_list,df_all_images,new_height,new_width):
    """Function to resize all images in a dataset to a new square size
    It will create a new directory containing all the resized images accordingly
    
    Parameters
    ----------
    path_to_dataset : str
        Path (absolute or relative) to contents of image dataset folder. 
        The dataset is assumed to have the following structure
        dataset/
            folder1/
                subfolder1.1/
            folder2/
                subfolder2.1/
                subfolder2.2/
            folder3/
    
    dir_list : list[str]
        A list of directories generated with `all_subdir_list`

    df_all_images : dataframe
        A dataframe listing all the images in a datasaet to be resized
        It is assumed to have at least the following columns: `item`, `label`,`channels`,`height`,`width`
    
    new_height : int
        Resized image height in pixels
    
    new_width : int
        Resized image height in pixels
    
    """
    assert isinstance(path_to_dataset,str),'path_to_dataset should be a string'
    assert isinstance(new_height,int) and new_height > 0,'new_height should be a positive integer'
    assert isinstance(new_width,int) and new_width > 0,'new_width should be a postive integer'
    
    # save original directory
    original_dir = os.getcwd()

    # change to dataset directory
    os.chdir(path_to_dataset)

    # get dataset folder name
    curr_wd = os.getcwd()
    folder_name = curr_wd.split('/')[-1]

    new_folder = f'../{folder_name}_{new_height}x{new_width}'

    for subdir in dir_list:

        print('\nWorking on:\t',subdir)
        
        path_new_subdir = os.path.join(new_folder,subdir)
        
        # make new subdirectory if it doesn't already exist
        if not os.path.exists(path_new_subdir):
            os.makedirs(path_new_subdir)

        # filter images by subdirectory
        record_filter = df_all_images['item'].str.contains(subdir)
        df_filtered = df_all_images[record_filter]

        # temporary list for all renamed images
        temp_img_list = []
        
        print('Updating images')
        
        for path_image in df_filtered['item']:

            # read image in dataset directory
            original_img = read_image(path_image)

            # Resize image
            temp = v2.Resize(size = (new_height,new_width))(original_img)

            # update image name from `name.extension` 
            # to 'name_{new_height}x{new_width}_.extension'
        
            # image extension (.png, .jpg)
            extension = path_image[-4:]
            
            # path + name of image 
            path_image_name = path_image[0:-4]

            path_new_image_name =f'{path_image_name}_{new_height}x{new_width}_{extension}'
            
            # save image in original_dataset/subdir/
            save_image(temp/255,path_new_image_name)

            temp_img_list.append(path_new_image_name)
            
        print(f'Moving resized images to {path_new_subdir}')

        for img in temp_img_list:
            shutil.move(img,path_new_subdir)

    # back to original directory
    os.chdir(original_dir)