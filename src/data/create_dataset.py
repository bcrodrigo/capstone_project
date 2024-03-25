import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomFireImagesDataset(Dataset):
    """Class to create a custom fire image dataset
    
    Attributes
    ----------
    directory : str
        Path (relative or absolute) to dataset directory, containing all images
    labels : integer
        indicates the class of the image
    len : integer
        length of the labels obtained in a batch
    transform : object
        transformation to be applied to the dataset
    """
    
    def __init__(self,annotations_file,directory,transform = None):
        """Constructor for CustomFireImagesDataset class
        
        Parameters
        ----------
        annotations_file : string
            Filename (.csv) containing the image names and labels
        directory : string
            Path (relative or absolute) to dataset directory, containing all images
        transform : None, optional
            Transformation to be applied to images. Default is None.
        """
        
        self.directory = directory
        
        annotations_file_dir = os.path.join(self.directory, annotations_file)
        
        self.labels = pd.read_csv(annotations_file_dir)
        
        # transform to be applied on images
        self.transform = transform

        # Number of images in dataset
        self.len = self.labels.shape[0]


    def __len__(self):
        """Length Method for CustomFireImagesDataset
        
        Returns
        -------
        Integer
            Length of the labels file
        """
        return len(self.labels)


    def __getitem__(self,idx):
        """Get item method for CustomFireImagesDataset
        
        Parameters
        ----------
        idx : integer
            Index of the image to be taken from the annotations file
        
        Returns
        -------
        Tuple
            Image (Pytorch tensor) and class label (integer)
        
        """
                
        image_path = os.path.join(self.directory, self.labels.iloc[idx, 0])
        
        # reading the images
        image = read_image(image_path)
        
        # labels of the images 
        label = self.labels.iloc[idx, 1]

        # apply the transform if not set to None
        if self.transform:
            image = self.transform(image)
        
        return image, label