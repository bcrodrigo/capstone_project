# to make plots related to model evaluation
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import roc_curve, roc_auc_score

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


def calculate_conf_matrix(label_list,pred_list,model_name,img_classes,savefig = False):
    """Function to calculate the Confusion Matrix and the classification
    report. 
    
    Parameters
    ----------
    label_list : list or array
        True labels
    pred_list : list or array
        Predictions from the model
    model_name : string
        Name of the model being evaluated
    img_classes : list
        Classes of the classification problem. Preferably a list of strings.
    savefig : bool, optional
        To save figure (confusion matrix) or not
    
    Returns
    -------
    string
        Classification Report
    """

    # calculate the confusion matrix 
    cm = confusion_matrix(label_list, pred_list)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = img_classes)

    fig, ax = plt.subplots(figsize = (5,4))
    disp = disp.plot(ax=ax, cmap = 'Blues')
    plt.tight_layout()
    plt.title(f'Confusion Matrix for {model_name}')
    
    if savefig:
        plt.savefig(f'confusion_matrix_{model_name}.png')
    
    plt.show()

    report = classification_report(label_list,pred_list)
    print(report)

    return report

def plot_roc_curve(label_list,pred_list,model_name, savefig = False):
    """Function to plot the ROC curve. It also calculates the AUC score.
    
    Parameters
    ----------
    label_list : list or array
        True labels
    pred_list : list or array
        Predictions from the model
    model_name : string
        Name of the model being evaluated
    savefig : bool, optional
        To save figure (ROC curve with AUC score) or not
    
    Returns
    -------
    number
        The calculated AUC score.
    """

    fpr, tpr, _ = roc_curve(label_list,pred_list)
    score = roc_auc_score(label_list,pred_list)

    plt.figure()
    plt.plot(fpr,tpr, color = 'b', lw = 1, label = f'AUC {score:0.4f}')
    plt.plot([0, 1], [0, 1], color='k', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.grid()
    plt.title(f'ROC for {model_name}')
    plt.legend()

    if savefig:
        plt.savefig(f'roc_auc_score_{model_name}.png')

    plt.show()

    return score