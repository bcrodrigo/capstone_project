import numpy as np
import matplotlib.pyplot as plt

def plot_loss_acc(train_losses, valid_losses,train_acc,valid_acc):
    '''Function for plotting training and validation losses and accuracies
    
    Parameters
    ----------
    train_losses : list or array
        Contains the Training Losses for each epoch
    valid_losses : list or array
        Contains the Validation Losses for each epoch
    train_acc : list or array
        Contains the Training Accuracies for each epoch
    valid_acc : list or array
        Contains the Validation Accuracies for each epoch
    '''

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    train_acc = np.array(train_acc)
    valid_acc = np.array(valid_acc)

    N = len(train_losses)
    epochs = np.arange(N)

    plt.subplots(1,2,figsize = (14,5))

    # plot losses
    plt.subplot(1,2,1)

    plt.plot(train_losses, color='blue', label='Training Loss') 
    plt.plot(valid_losses, color='red', label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.xticks(ticks = epochs)
    plt.xlim(min(epochs),max(epochs))
    plt.legend()
    plt.grid()

    # plot accuracies
    plt.subplot(1,2,2)

    plt.plot(train_acc, color='blue', label='Training Accuracy') 
    plt.plot(valid_acc, color='red', label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xticks(ticks = epochs)
    plt.xlim(min(epochs),max(epochs))
    plt.legend()
    plt.grid()
    
    plt.show()