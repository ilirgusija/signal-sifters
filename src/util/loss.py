#!/usr/bin/env python3

import torch
import numpy as np
import abc
import sys
import torch.nn as nn


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
    errors = np.sqrt(errorvectors[:, 0]**2 + errorvectors[:, 1]**2)
    mae = np.mean(errors)
    r90 = np.percentile(errors, 90)

    return mae, r90

##########################
# Channel Charting Loss #
##########################


class ChannelChartingLoss(nn.Module):
    def __init__(self, timestamps, acceleration_mean=0.0, acceleration_variance=1.7, acceleration_weight=0.01):
        super().__init__()
        self.register_buffer('timestamps', timestamps.detach().clone())

        assert acceleration_mean == 0.0, "Only zero-mean acceleration is supported."

        self.acceleration_mean = acceleration_mean
        self.acceleration_variance = acceleration_variance
        self.acceleration_weight = acceleration_weight

    def acceleration(self, pred_positions):
        # pred_positions: (N, D)
        dt = torch.diff(self.timestamps)
        pred_velocities = torch.diff(
            pred_positions, dim=0) / dt.unsqueeze(-1)  # (N-1, D)
        pred_accelerations = torch.diff(
            pred_velocities, dim=0) / dt[:-1].unsqueeze(-1)  # (N-2, D)
        pred_accelerations_abs = torch.sqrt(
            torch.sum(
                pred_accelerations**2, dim=-1
            ) + 1e-8
        )  # (N-2,)

        # This is the "folded normal distribution model" described in the paper that we would ideally like to use.
        # Problem: ln(exp() + exp()) is numerically not nice...
        # folded_a = -tf.square(pred_accelerations_abs - self.acceleration_mean) / (2 * self.acceleration_variance)
        # folded_b = -tf.square(pred_accelerations_abs + self.acceleration_mean) / (2 * self.acceleration_variance)
        # return -tf.math.reduce_mean(tf.math.log(tf.math.exp(folded_a) + tf.math.exp(folded_b) + 1e-25))

        # Therefore, we use a simpler model that only supports self.acceleration_mean = 0 (which is guaranteed by assertion in __init__).
        # In that case (zero-mean folded normal distribution), ln and exp cancel out nicely:
        return torch.mean(torch.square(pred_accelerations_abs) / self.acceleration_variance)

    def forward(self, y_true, y_pred):
        # This is an ugly workaround, the loss function always gets y_pred as float, convert back to integer for index
        # This works as long as CSI tensor is not absolutely huge (16M+ entries), which can be assumed.
        # y_true: (B, 3+longest_shortest_path), y_pred: (N, D)
        # y_true: [path_hops, path_means, path_variances, paths...] (B, 3+longest_shortest_path)
        assert not torch.isnan(y_true).any(), "y_true is NaN"
        assert not torch.isnan(y_pred).any(), "y_pred is NaN"
        y_true = y_true.squeeze(0)
        path_hops = y_true[:, 0].long()
        path_means = y_true[:, 1]
        path_variances = y_true[:, 2]
        paths = y_true[:, 3:].long()  # (B, longest_shortest_path)

        path_end_indices = torch.transpose(
            torch.stack(
                [torch.arange(y_true.shape[0]), path_hops]
            ), 0, 1
        )
        longest_shortest_path = paths.shape[1]
        # if path_hops contains longest_shortest_path, set it to 0
        path_hops = torch.where(path_hops >= longest_shortest_path, 0, path_hops)
        index_A = y_true[:, 3].long()
        try:
            index_B = paths[torch.arange(y_true.shape[0]), path_hops].long()
        except IndexError as e:
            # print original error message
            print(f"IndexError: {e}")
            print(f"paths.shape: {paths.shape}")
            print(f"torch.arange(paths.shape[0]).shape: {torch.arange(paths.shape[0]).shape}")
            print(f"path_hops.shape: {path_hops.shape}")
            print(f"y_true.shape: {y_true.shape}")
            print(f"index_A.shape: {index_A.shape}")
            print(f"path_end_indices.shape: {path_end_indices.shape}")
            print(f"path_end_indices[:,0].shape: {path_end_indices[:,0].shape}")
            print(f"path_end_indices[:,1].shape: {path_end_indices[:,1].shape}")
            raise IndexError

        pos_A = y_pred[index_A]
        pos_B = y_pred[index_B]

        # Acceleration loss (not applied during pre-training phase)
        acceleration_loss = self.acceleration(y_pred)

        # Geodesic loss
        # paths has shape (BATCHSIZE, longest_shorest_path)
        # path_positions has shape (BATCHSIZE, longest_shorest_path, 2), where last dimension is x/y position
        # path_positions_delta has shape (BATCHSIZE, longest_shorest_path - 1, 2), where last dimension is x/y delta
        # path_distances has shape (BATCHSIZE, longest_shorest_path - 1)
        # endpoint_distances has shape (BATCHSIZE)
        path_positions = y_pred[paths]  # (B, longest_shortest_path, D)
        path_positions_delta = path_positions[:,
                                              1:, :] - path_positions[:, :-1, :]
        path_distances = torch.sqrt(
            torch.sum(
                path_positions_delta**2, dim=-1) + 1e-6)
        geodesic_distance = path_distances.sum(dim=1)
        geodesic_loss = torch.mean(
            torch.square(geodesic_distance - path_means) / (path_variances + 1e-6))

        # Make sure all path distances are smaller than the endpoint distances
        endpoint_distances = torch.sqrt(
            torch.sum(
                torch.square(pos_A - pos_B), dim=1))
        geodesic_loss = geodesic_loss + 0.01 * \
            torch.sum(torch.clamp(
                path_distances - endpoint_distances.unsqueeze(1), min=0)
            )
        
        assert not torch.isnan(acceleration_loss), "Acceleration loss is NaN"
        assert not torch.isnan(geodesic_loss), "Geodesic loss is NaN"

        # Combination
        loss = geodesic_loss + self.acceleration_weight * acceleration_loss
        return loss
