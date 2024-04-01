import numpy as np
import matplotlib.pyplot as plt

def plot_loss_acc(history):
    '''Function for plotting training and validation losses and accuracies
    
    Parameters
    ----------
    history : dictionary
        This is one of the outputs from `train_and_validate()`
        Contains the following keys 
        - train_losses
        - train_accuracy
        - validation_losses
        - validation_accuracy
    '''

    N = len(history['train_losses'])
    epochs = np.arange(N)

    plt.figure()

    plt.plot(history['train_losses'], color='blue', label='Training Loss') 
    plt.plot(history['validation_losses'], color='red', label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    
    plt.xticks(ticks = epochs)
    plt.xlim(min(epochs),max(epochs))
    plt.legend()
    plt.grid()

    plt.show()
    
    # plot accuracies
    plt.figure()

    plt.plot(history['train_accuracy'], color='blue', label='Training Accuracy') 
    plt.plot(history['validation_accuracy'], color='red', label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    
    plt.xticks(ticks = epochs)
    plt.xlim(min(epochs),max(epochs))
    plt.legend()
    plt.grid()
    
    plt.show()