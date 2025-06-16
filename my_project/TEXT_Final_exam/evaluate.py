import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib (for GUI support)


def evaluate_model(model, dataloader, device):
    # Set the model to evaluation mode (turns off dropout, etc.)
    model.eval()
    preds, targets = [], []  # Lists to store predictions and true labels

    # Disable gradient calculations (to save memory and computations during evaluation)
    with torch.no_grad():
        for batch in dataloader:
            # Load the input data (text and image features) from the batch and move them to the specified device (e.g., GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            # Get model predictions (outputs)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=images)
            pred = torch.argmax(outputs, dim=1)  # Get the class with the highest probability

            # Append predictions and true labels to their respective lists
            preds.extend(pred.cpu().numpy())  # Move predictions to CPU and convert to numpy
            targets.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy

    # Calculate various performance metrics
    acc = accuracy_score(targets, preds)  # Accuracy
    prec = precision_score(targets, preds, average='weighted')  # Precision (weighted average)
    rec = recall_score(targets, preds, average='weighted')  # Recall (weighted average)
    f1 = f1_score(targets, preds, average='weighted')  # F1 score (weighted average)

    # Print the evaluation metrics
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")

    # Generate and display the confusion matrix
    cm = confusion_matrix(targets, preds)  # Compute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Display confusion matrix as a heatmap
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
