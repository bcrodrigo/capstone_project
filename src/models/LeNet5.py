import torch
import torch.nn as nn

class LeNet5(nn.Module):
    '''
    LeNet-5 Image classifier for images sized [3,32,32] and arbitrary number of classes.
    
    Note the activation functions have been modified to ReLU Pooling Layers are MaxPool
    
    '''
    def __init__(self, n_classes):
        """Constructor for LeNet5 class
        
        Parameters
        ----------
        n_classes : Integer
            Number of classes for the classification task. It has to be at least 2.
        """

        assert isinstance(n_classes,int),'n_classes must be an integer'
        assert (n_classes>=2),'n_classes must be an integer >= 2'

        super(LeNet5, self).__init__()
        
        self.convolutional_layers = nn.Sequential(            
            # First Convolution
            # Input Image size 3 x 32 x 32
            # Output is 6 x 28 x 28
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1),
            nn.ReLU(),

            # Pool
            # Input is 6 x 28 x 28
            # Output is 6 x 14 x 14
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            # Second Convolution
            # Input Image size 6 x 14 x 14
            # Output is 16 x 10 x 10
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            nn.ReLU(),

            # Second Pool
            # Input is 16 x 10 x 10
            # Output is 16 x 5 x 5
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            
            # Third Convolution
            # Input is 16 x 5 x 5
            # Output is 120 x 1 x 1
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 84),
            nn.ReLU(),
            nn.Linear(in_features = 84, out_features = n_classes),
        )

    def forward(self, x):
        """Forward pass method for LeNet5 class
        
        Parameters
        ----------
        x : Tensor
            Image tensor of size [batch_size,3,32,32]
        
        Returns
        -------
        Tensor
            Un-normalized logits after forward pass, note that
            the output is a tensor of size [batch_size,n_classes]
        """
        x = self.convolutional_layers(x)
        
        # Make x a one-dimentional tensor of size 120
        x = torch.flatten(x, 1)

        x = self.fc_layers(x)
        
        # Return un-normalized logits
        return x

    def predict(self,x):
        '''Method to get hard class predictions
        
        Parameters
        ----------
        x : Tensor
            Image tensor of size [batch_size,3,32,32]
        
        Returns
        -------
        Tensor
            Tensor with the highest class logit, note the output size
            is a sized [batch_size,1], where each column contains the 
            hard class prediction (integer between 0 and n_classes)

        '''
        predictions = self.forward(x)
        
        # Find highest class logit
        hard_class_predictions = torch.argmax(predictions, dim=1)
        
        return hard_class_predictions

    def predict_proba(self,x):
        """Method to get the soft predictions (probabilities) of each class
        
        Parameters
        ----------
        x : Tensor
            Image tensor of size [batch_size,3,32,32]
        
        Returns
        -------
        Tensor
            Predicted probabilities for each class. The output size
            is [bach_size,n_classes]
        """

        # First get the un-normalized logits
        predictions = self.forward(x)
        
        # Instantiate Softmax along dimension 1
        softmax_layer = nn.Softmax(dim = 1)

        # Calculate probabilities
        predicted_prob = self.softmax_layer(predictions)

        return predicted_prob