# will contain all functions needed to instantiate models for transfer learning

from torchvision import models
import torch.nn as nn


def vgg19_transfer_learning(N_CLASSES):
    """Function to create an instance of VGG19 suitable for transfer learning 
    of classification tasks
    
    Parameters
    ----------
    N_CLASSES : integer
        Number of classes for the classification task. Must be >= 2
    
    Returns
    -------
    Object
        PyTorch model (VGG19) ready to be trained on the last layer only
    """
    assert isinstance(N_CLASSES,int),'N_CLASSES must be an integer'
    assert (N_CLASSES >= 2),'N_CLASSES must be >= 2'

    # download weights
    vgg19 = models.vgg19(weights = 'DEFAULT')

    # freeze model parameters for training
    for param in vgg19.parameters():
        param.requires_grad = False

    # update the last layer to output N_CLASSES
    vgg19.classifier[6] = nn.Linear(in_features = 4096, out_features = N_CLASSES, bias = True)

    return vgg19