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


def ai_slop(positions_estimated, position_labels):
    # positions_estimated, position_labels: (batch_size, 2) or (batch_size, 3)
    errorvectors = position_labels - positions_estimated
    errors = torch.sqrt(errorvectors[:, 0] ** 2 + errorvectors[:, 1] ** 2)
    mae = torch.mean(errors)
    # r90 is not differentiable due to percentile, so we return only mae for differentiability
    return mae

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