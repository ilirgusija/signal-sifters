# from src.util.dataset import DichasusDataset, ChannelChartingPathDataset
# from src.util.loss import ChannelChartingLoss, ADPDissimilarityMetric, VelocityDissimilarityMetric
# from src.models.gaussian_dissimilarity import GaussianDissimilarityModel
# from src.models.channel_charting import ChannelChart
import torch
import sys
import os
# from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train(model, dataloader, optimizer, scheduler, loss_fn, epochs=1, lr=1e-2, lr_final=1e-4, device='cpu', plot_callback=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def lr_lambda(epoch): return (
        lr_final / lr) ** (epoch / max(epochs-1, 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda)
    history = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0
        for batch_idx, (indices, y_true) in enumerate(tqdm(dataloader, desc="Batches", leave=False)):
            indices, y_true = indices.long().to(device), y_true.to(device)
            optimizer.zero_grad()

            # Forward pass
            y_pred = model.forward(indices)

            # Call metric to update callback state
            if plot_callback is not None:
                plot_callback.metric(y_true, y_pred)
            loss = loss_fn(y_true, y_pred)
            # Terminate on NaN
            if torch.isnan(loss):
                print(
                    f"NaN loss encountered at epoch {epoch}, batch {batch_idx}. Stopping training.")
                if plot_callback is not None:
                    plot_callback.on_train_end()
                return history

            # Backward pass
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Call batch-end callbacks
            if plot_callback is not None:
                plot_callback.on_train_batch_end(
                    batch_idx, logs={"loss": loss.item()})
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        history.append(avg_loss)

    # Call train-end callbacks
    if plot_callback is not None:
        plot_callback.on_train_end()
    return history


EPOCHS = 10
MIN_BATCH_SIZE = 500
MAX_BATCH_SIZE = 4000
LEARNING_RATE_INITIAL = 1.5e-2
LEARNING_RATE_FINAL = 1e-2
MAX_HOPLENGTH = 20.0
MIN_HOPLENGTH = 0.1
RANDOMIZE_PATHHOPS = False
PLOT_CALLBACK = None
TRAINING_BATCHES = 3000
ACCELERATION_MEAN = 0.0
ACCELERATION_VARIANCE = 0.5
ACCELERATION_WEIGHT = 1e-4
MAX_PATHHOPS = 100

# if __name__ == "__main__":
#     device = 'cpu'
#     pre_processed_data = DichasusDataset("labeled_tdomain.npz", device=device)

#     # 1. Extract tensors
#     csi_time_domain = pre_processed_data.csi  # [N, ...]
#     groundtruth_positions = pre_processed_data.pos  # [N, ...]
#     timestamps = pre_processed_data.time  # [N]

#     # 2. Process timestamps
#     MEASUREMENT_INTERVAL = 0.048
#     timestamps = timestamps - timestamps[0]
#     timestamps = torch.round(
#         timestamps / MEASUREMENT_INTERVAL) * MEASUREMENT_INTERVAL

#     adp_metric = ADPDissimilarityMetric(csi_time_domain)

#     velocity_mean = 0.235
#     velocity_variance = 0.010

#     # Make worst-case assumption of perfectly correlated velocities. This maximizes the variance.
#     velocity_metric = VelocityDissimilarityMetric(
#         velocity_mean, velocity_variance, True, timestamps)

#     gdm = GaussianDissimilarityModel([adp_metric, velocity_metric])
#     # TODO: This function call takes 30+ minutes to run. We should pre-generate the paths and save them to a file.
#     gdm.generate_short_paths(total_path_count=40000, realization_count=8)

#     # TODO: This may also be an issue, dependent on how long get_random_short_paths takes. This could be an alternative target for serialization.
#     training_dataset = ChannelChartingPathDataset(
#         gdm,
#         csi_time_domain,
#         training_batches=TRAINING_BATCHES,
#         max_pathhops=MAX_PATHHOPS,
#         min_batch_size=MIN_BATCH_SIZE,
#         max_batch_size=MAX_BATCH_SIZE,
#         min_hoplength=MIN_HOPLENGTH,
#         max_hoplength=MAX_HOPLENGTH,
#         randomize_pathhops=RANDOMIZE_PATHHOPS,
#         device=device
#     )

#     # For now, we set this to 1 since we are doing most of the heavy lifting in the dataset class
#     dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
#     model = ChannelChart(csi_time_domain)
#     loss_fn = ChannelChartingLoss(
#         timestamps,
#         acceleration_mean=ACCELERATION_MEAN,
#         acceleration_variance=ACCELERATION_VARIANCE,
#         acceleration_weight=ACCELERATION_WEIGHT
#     )
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_INITIAL)
#     learning_rate_decay_factor = LEARNING_RATE_FINAL / LEARNING_RATE_INITIAL
#     scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optimizer, lr_lambda=lambda epoch: learning_rate_decay_factor ** (
#             epoch / max(EPOCHS-1, 1))
#     )
#     train(model, dataloader, optimizer, scheduler,
#           epochs=EPOCHS, device='cpu', plot_callback=None)
