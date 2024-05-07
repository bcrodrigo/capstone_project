# functions to make predictions
import torch

def make_hard_predictions(trained_model,test_dataloader,DEVICE):
    '''Function to make hard predictions with a trained PyTorch model.
    
    Parameters
    ----------
    trained_model : object
        Trained model to be used for inference. It is assumed the model by default
        outputs the raw logits
    
    test_dataloader : dataloader
        Dataloader for the test images to be used for inference. Please ensure it
        doesn't shuffle the images, that way it will be easiest to retrieve from the original
        annotations file
    DEVICE : string
        Device that will perform the calculations: cuda, mps, or cpu
    
    Returns
    -------
    tuple
        true label list (from dataloader)
        predicted label list (hard predictions from model)
    '''

    label_list = []
    pred_list = []

    # transfer model to selected device
    trained_model = trained_model.to(DEVICE)

    # Set model to evaluation mode 
    trained_model.eval()
    
    for batch in test_dataloader:
        
        # Get images and labels from batch
        images, true_labels = batch
        
        # transfer images and labels to selected device
        images = images.to(DEVICE)
        true_labels = true_labels.to(DEVICE)

        pred_logits = trained_model(images)

        # Get predicted labels (hard predictions) on images
        # currently the model implementation doesn't account for this
        pred_labels = torch.argmax(pred_logits, dim = 1)

        label_list.extend(list(true_labels.cpu().numpy()))
        pred_list.extend(list(pred_labels.cpu().numpy()))

    return label_list,pred_list