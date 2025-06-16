import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        """
        Initialize the FocalLoss module.

        Focal Loss is used to address class imbalance by focusing on hard-to-classify examples
        and down-weighting easy examples.

        Parameters:
        - gamma (float): Focusing parameter to control the rate at which easy examples are down-weighted.
        - alpha (float): Balancing factor to weight the importance of the positive class.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # The focusing parameter to control the down-weighting of easy examples
        self.alpha = alpha  # The balancing parameter for handling class imbalance

    def forward(self, logits, targets):
        """
        Forward pass to compute the Focal Loss.

        Parameters:
        - logits (Tensor): Predicted logits (raw scores from the model).
        - targets (Tensor): True class labels.

        Returns:
        - Tensor: The calculated focal loss averaged over the batch.
        """
        # Calculate the standard Cross-Entropy loss without reduction (to get per-sample loss)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Calculate the probability for the correct class using the exponential of negative CE loss
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class

        # Compute the focal loss using the modulating factor (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Return the mean of the focal loss over the batch
        return focal_loss.mean()
