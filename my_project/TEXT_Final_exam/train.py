import torch
from tqdm import tqdm
from losses import FocalLoss  # Importing the custom Focal Loss function


def train_model(model, dataloader, optimizer, device):
    model.train()  # Set the model to training mode (enables features like dropout)
    criterion = FocalLoss()  # Initialize the Focal Loss function
    total_loss = 0  # Variable to accumulate the total loss for the epoch

    # Initialize tqdm for progress bar during training
    progress = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress:
        # Move the batch data to the correct device (GPU or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # Zero out gradients for the optimizer before backpropagation
        optimizer.zero_grad()

        # Forward pass: compute model outputs given the input data
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)

        # Calculate the loss using the model outputs and true labels
        loss = criterion(outputs, labels)

        # Backpropagate the loss to compute gradients
        loss.backward()

        # Update the model parameters based on the gradients
        optimizer.step()

        # Accumulate the total loss for the epoch
        total_loss += loss.item()

        # Update the progress bar with the current loss
        progress.set_postfix(loss=loss.item())

    # Return the average loss over the entire epoch
    return total_loss / len(dataloader)
