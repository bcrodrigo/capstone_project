import torch
import time
from datetime import datetime

def train_and_validate(model,loss_function,optimizer,N_EPOCHS,train_dataloader,val_dataloader):
    """Function to run training and validation loops for a given model
    
    Parameters
    ----------
    model : class
        Model to be trained
    loss_function : class
        Instance of a loss function
    optimizer : class
        Instance of optimizer to be used
    N_EPOCHS : integer
        Number of epochs to be used for training
    train_dataloader : dataloader
        Training set dataloader
    val_dataloader : dataloader
        Validation set dataloader
    
    Returns
    -------
    tuple
        model after training
        history dictionary containing 
            train loss
            train accuracy 
            validation loss
            validation accuracy
    """
    assert isinstance(N_EPOCHS,int),'N_EPOCHS must be an integer'
    assert (N_EPOCHS >= 0),'N_EPOCHS must be positive'

    # dictionary to accumulate losses and accuracies
    history = {
        'train_losses':[],'train_accuracy':[],
        'validation_losses':[],'validation_accuracy':[]
    }

    # ---- to be implemented ----
    # start = time.time()

    for epoch in range(N_EPOCHS):

        # Set to training mode
        model.train()

        # Loss and Accuracy within the epoch
        train_loss_epoch = 0.0
        train_acc_epoch = 0.0

        valid_loss_epoch = 0.0
        valid_acc_epoch = 0.0

        # Training loop 
        for batch in train_dataloader:
            
            # Get images and labels from batch
            images, true_labels = batch

            # --- to be implemented ---
            # images = images.to(device)
            # true_labels = true_labels.to(device)

            # make gradients zero
            optimizer.zero_grad()

            # Forward Pass
            pred_logits = model(images)
            
            # Loss Calculation
            loss = loss_function(pred_logits,true_labels)

            # Backpropagate gradient
            loss.backward()
            
            # Update model weights
            optimizer.step()
            
            # Accumulate loss for this batch
            train_loss_epoch += loss.item()

            # Get predicted labels (hard predictions) on images
            # currently the implementation doesn't account for this
            pred_labels = torch.argmax(model(images), dim = 1)

            train_acc_epoch += (true_labels == pred_labels).sum().item() / true_labels.shape[0]


        # Calculate average training loss and training accuracy
        average_loss = train_loss_epoch / len(train_dataloader)
        average_accuracy = train_acc_epoch / len(train_dataloader)

        # Update dictionary with training losses and accuracy for 
        history['train_losses'].append(average_loss)
        history['train_accuracy'].append(average_accuracy)

        
        # Start validation -  no gradient tracking needed
        with torch.no_grad():

            # Set model to evaluation mode
            model.eval()

            for batch in val_dataloader:

                images, true_labels = batch

                # ------ to be implemented --------
                # images = images.to(device)
                # true_labels = true_labels.to(device)

                # forward pass
                pred_logits = model(images)

                # compute loss
                loss = loss_function(pred_logits,true_labels)

                # Accumulate loss for this batch
                valid_loss_epoch += loss.item()

                # Get predicted labels (hard predictions) on images
                # currently the model implementation doesn't account for this
                pred_labels = torch.argmax(pred_logits, dim = 1)

                valid_acc_epoch += (true_labels == pred_labels).sum().item() / true_labels.shape[0]

        # Calculate average validation loss and accuracy
        avg_validation_loss = valid_loss_epoch / len(val_dataloader)
        avg_validation_acc = valid_acc_epoch / len(val_dataloader)

        # Update dictionary with validation losses and accuracy
        history['validation_losses'].append(avg_validation_loss)
        history['validation_accuracy'].append(avg_validation_acc)
        
        print(f'{datetime.now().time().replace(microsecond=0)}\t'
              f'Epoch: {epoch+1}/{N_EPOCHS}\t'
              f'Train loss: {average_loss:.4f}\t'
              f'Val. loss: {avg_validation_loss:.4f}\t'
              f'Train acc.: {100 * average_accuracy:.2f}\t'
              f'Val. acc.: {100 * avg_validation_acc:.2f}')

    print('Finished Training') 

    return model, history
    

def get_loss_acc(model,loss_func, dataloader):
    '''Helper function to get validation losses and classification accuracy
    over the items in dataloader.
    
    Parameters
    ----------
    model : PyTorch model
        Description
    loss_func : TYPE
        Description
    dataloader : TYPE
        Description
    
    Returns
    -------
    floating point
        validation accuracy
        validation loss
    
    '''
    
    curr_loss = 0.0
    correct = 0

    # Go through all of the data
    for batch in dataloader:
    
        images, true_labels = batch
        
        # Get the hard prediction
        pred_labels = model.predict(images)
        
        # Get the logits
        pred_logits = model(images)

        # Calculate loss
        loss = loss_func(pred_logits,true_labels)

        # Accumulate current loss
        curr_loss += loss.item()

        # Count number of correct predictions
        correct += (pred_labels == true_labels).sum().item() / true_labels.shape[0]
        
    validation_accuracy = correct / len(dataloader)
    validation_loss = curr_loss / len(dataloader)
    
    return validation_accuracy,validation_loss