import torch
import torch.nn.functional as F


def binary_top_k_loss(y_true, y_pred, k=100):
    """
    Computes a binary cross-entropy loss focused on predicting whether genes are in the top-k most changed.

    Parameters:
    - y_true (torch.Tensor): Ground truth gene expression changes, shape (batch_size, gene_count).
    - y_pred (torch.Tensor): Predicted gene expression changes, shape (batch_size, gene_count).
    - k (int): Number of top genes to focus on.

    Returns:
    - torch.Tensor: Binary cross-entropy loss for top-k prediction.
    """
    # Identify top-k indices based on true values
    top_k_indices = torch.argsort(torch.abs(y_true), dim=1, descending=True)[:, :k]

    # Create a binary label tensor (1 for top-k, 0 otherwise)
    binary_labels = torch.zeros_like(y_true)
    for i in range(y_true.size(0)):
        binary_labels[i, top_k_indices[i]] = 1

    # Apply sigmoid to predicted values
    # y_pred_prob = torch.sigmoid(y_pred)

    # Calculate binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, binary_labels)

    return bce_loss
