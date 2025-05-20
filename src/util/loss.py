#!/usr/bin/env python3

import torch
import numpy as np
import sys


# \alpha*mean + \mbb{E}(1-\alpha)e^{beta*error^2}
def train_loss(positions_estimated, position_labels, alpha=0.5, beta=10.0):
    # print(positions_estimated.shape)
    # print(position_labels.shape)

    # sys.exit()
    # cut off last dimension of labels
    position_labels = position_labels[:, :2]

    errorvectors = position_labels - positions_estimated
    errors = torch.sqrt(errorvectors[:, 0] ** 2 + errorvectors[:, 1] ** 2)
    return alpha * torch.mean(errors) + (1 - alpha) * torch.mean(torch.exp(beta * (errors ** 2)))


def differentiable_r90(errors):
    """
    Differentiable approximation of R90 using soft ranking.
    Instead of taking a hard 90th percentile, we use a soft approximation
    that considers all errors with appropriate weighting.
    """
    # Sort errors
    sorted_errors, _ = torch.sort(errors)
    batch_size = errors.shape[0]
    
    # Calculate the index that would represent the 90th percentile
    target_idx = int(0.9 * batch_size)
    
    # Create weights that peak around the 90th percentile
    # Using a soft approximation with temperature
    temperature = 10.0
    indices = torch.arange(batch_size, device=errors.device)
    weights = torch.exp(-temperature * torch.abs(indices - target_idx))
    weights = weights / weights.sum()  # normalize weights
    
    # Compute weighted sum of errors as R90 approximation
    r90_approx = (sorted_errors * weights).sum()
    
    return r90_approx


def ai_slop(positions_estimated, position_labels, alpha=0.5):
    """
    Combined loss function using both MAE and a differentiable R90 approximation
    """
    # positions_estimated, position_labels: (batch_size, 2) or (batch_size, 3)
    errorvectors = position_labels - positions_estimated
    errors = torch.sqrt(errorvectors[:, 0] ** 2 + errorvectors[:, 1] ** 2)
    
    # Calculate MAE
    mae = torch.mean(errors)
    
    # Calculate differentiable R90
    r90 = differentiable_r90(errors)
    
    # Combine both metrics with equal weighting
    combined_loss = alpha * mae + (1 - alpha) * r90
    
    return combined_loss

# Function that was the old ai_slop (just the MAE):
def old_ai_slop(positions_estimated, position_labels):
    errorvectors = position_labels - positions_estimated
    errors = torch.sqrt(errorvectors[:, 0] ** 2 + errorvectors[:, 1] ** 2)
    return torch.mean(errors)

# ===============================
# ===============================
# ===============================

def compute_localization_metrics(positions_estimated, position_labels):
    """
    (Competition provided function)
    Compute MAE and r90 from estimated and ground truth positions
    """
    errorvectors = position_labels - positions_estimated
    errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)
    mae = np.mean(errors)
    r90 = np.percentile(errors, 90)

    return mae, r90