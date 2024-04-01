# Contains functions to visualize images and tensors
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import make_grid

def display_tensor(tensor_name,label_name = ''):
    '''Function to display an image. The image is assumed to be a PyTorch RGB tensor (3,W,H)

    Parameters
    ----------
    tensor_name : Tensor
        PyTorch image tensor

	label_name : str
        Optional string with the image label to display.

    Returns
    -------
    None

    '''
    plt.title(label_name)
    plt.imshow(tensor_name.permute(1,2,0))
    plt.show()

def display_image(image_name, img_title = None): 
    '''Function to display an image. The image is assumed to be a 3-channel RGB .png or .jpg
    
    Parameters
    ----------
    image_name : str
        String with the path pointing to the image
    img_title : None, optional
        String with an optional title for the image. By default it assigns `image_name` as the title.
    '''
    
    if img_title is None:
        img_title = image_name

    img = read_image(image_name)
    plt.title(img_title)
    plt.imshow(img.permute(1,2,0))
    plt.show()


def display_image_batch(data_loader,n_batches,img_classes):
    """Function to display a batch of images from a dataloader
    
    Parameters
    ----------
    data_loader : object
        PyTorch dataloader object
    n_batches : integer
        Number of batches to display
    img_classes : tuple
        Image classes contained in the dataset

    """
    # First create an iterator
    dataiter = iter(data_loader)

    for k in range(n_batches):
        # Extract images and labels
        images,labels = next(dataiter)
        img = make_grid(images)

        plt.figure(figsize = (10,5))
        plt.imshow(img.permute(1,2,0))
        N = len(labels)
        print(' '.join('%5s' % img_classes[labels[j]] for j in range(N)))
        plt.show()