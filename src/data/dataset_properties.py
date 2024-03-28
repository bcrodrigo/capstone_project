import torch
from torch.utils.data import ConcatDataset


def dataset_stat_prop(dataset):
    """Function to calculate the statistical properties of a custom dataset.
    The intent is to use the calculated mean and std as an input to 
    `torchvision.transforms.Normalize()` prior to training a model
    
    Parameters
    ----------
    dataset : object
        Pytorch dataset object. Note that the images contained in the dataset 
        should be in the [0,1] interval.
    
    Returns
    -------
    Tuple
        mean and standard deviation of all images in dataset. 
    """
    
    # stack all images together into a tensor of shape 
    # (N_images, 3, Height, Width)
    # note `sample` is a tuple with (tensor, label)
    
    x = torch.stack([sample[0] for sample in ConcatDataset([dataset])])

    # Get mean and standard deviation. Note we're getting the statistical properties
    # along all images (dim 0), all heights (dim 2) and all widths (dim 3) leaving
    # resulting in a 1 x 3 tensor 
    
    mean = torch.mean(x, dim = (0,2,3)) 
    std = torch.std(x, dim = (0,2,3))

    return mean, std