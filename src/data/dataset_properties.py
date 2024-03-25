from torch.utils.data import ConcatDataset
import torch
import torchvision
import torchvision.transforms as transforms

def dataset_stat_prop(dataset):
    """Function to calculate the statistical properties of a custom dataset
    
    Parameters
    ----------
    dataset : object
        Pytorch dataset object
    
    Returns
    -------
    Tuple
        mean and standard deviation of all images in dataset
    """

    # Normalize the images to the [0,1] interval
    # transform = transforms.Compose([transforms.ToTensor()])

    # Make sure the dataset has no transform
    # dataset.transform = None
    
    # stack all images together into a tensor of shape 
    # (N_images, 3, Height, Width)
    # note `sample` is a tuple with (tensor, label)
    
    x = torch.stack([sample[0] for sample in ConcatDataset([dataset])])

    # convert to floating point, otherwise can't calculate the mean and std
    x = x.to(torch.float)

    # Get mean and standard deviation. Note we're getting the statistical properties
    # along all images (dim 0), all heights (dim 2) and all widths (dim 3) leaving
    # resulting in a 1 x 3 tensor 
    
    mean = torch.mean(x, dim = (0,2,3)) 
    std = torch.std(x, dim = (0,2,3))

    return mean, std