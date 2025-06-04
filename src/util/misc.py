import torch
import matplotlib.pyplot as plt


# Implement device detection
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class PlotChartCallback:
    def __init__(self, groundtruth_positions, datapoint_count, max_hops=100, paths_to_plot_count=50, update_period=200):
        self.y_true = None
        self.y_pred = None

        self.groundtruth_positions = groundtruth_positions
        self.datapoint_count = datapoint_count
        self.paths_to_plot_count = paths_to_plot_count
        self.max_hops = max_hops
        self.update_period = update_period

    def set_model(self, model):
        self.training_model = model
        self.y_true = torch.zeros(
            (self.paths_to_plot_count, 1 + 2 + self.max_hops + 1), dtype=torch.float32)
        self.y_pred = torch.zeros(
            (self.datapoint_count, 2), dtype=torch.float32)

    def metric(self, y_true, y_pred):
        self.y_true = y_true[:self.paths_to_plot_count].to(torch.float32)
        self.y_pred = y_pred
        return 0

    @staticmethod
    def plot_colorized(positions, groundtruth_positions, title=None, show=True, alpha=1.0):
        # Generate RGB colors for datapoints
        center_point = torch.zeros(2, dtype=torch.float32)
        center_point[0] = 0.5 * (torch.min(groundtruth_positions[:, 0]) +
                                 torch.max(groundtruth_positions[:, 0]))
        center_point[1] = 0.5 * (torch.min(groundtruth_positions[:, 1]) +
                                 torch.max(groundtruth_positions[:, 1]))

        def NormalizeData(in_data):
            return (in_data - torch.min(in_data)) / (torch.max(in_data) - torch.min(in_data))

        rgb_values = torch.zeros((groundtruth_positions.shape[0], 3))
        rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
        rgb_values[:, 1] = 0.8 * \
            NormalizeData(torch.square(torch.norm(
                groundtruth_positions - center_point, dim=1)))
        rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

        # Plot datapoints
        plt.figure(figsize=(6, 6))
        if title is not None:
            plt.title(title, fontsize=16)
        plt.scatter(
            positions[:, 0].cpu().numpy(),
            positions[:, 1].cpu().numpy(),
            c=rgb_values.cpu().numpy(),
            alpha=alpha, s=10, linewidths=0
        )
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        if show:
            plt.show()

    @staticmethod
    def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos):
        # Pad with ones for affine transform
        def pad(x): return torch.hstack(
            [x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)])

        def unpad(x): return x[:, :-1]
        A, _, _, _ = torch.linalg.lstsq(
            pad(channel_chart_pos), pad(groundtruth_pos))

        def transform(x): return unpad(torch.matmul(pad(x), A))
        return transform(channel_chart_pos)

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.update_period == self.update_period - 1:
            pred_positions = self.y_pred

            channel_chart_positions_transformed = self.affine_transform_channel_chart(
                self.groundtruth_positions, pred_positions
            )
            errorvectors = self.groundtruth_positions - channel_chart_positions_transformed
            errors = torch.sqrt(
                errorvectors[:, 0] ** 2 + errorvectors[:, 1] ** 2)
            mae = torch.mean(errors).item()
            cep = torch.median(errors).item()

            self.plot_colorized(
                pred_positions, self.groundtruth_positions,
                title=f"MAE = {mae:.4f}m, CEP = {cep:.4f}m", show=False
            )

            y_true_np = self.y_true.to(torch.int32).cpu().numpy()
            paths_indices = y_true_np[:, 3:]

            for path_indices in paths_indices:
                # Remove any padding (e.g., repeated last index)
                path_indices = path_indices[path_indices >= 0]
                path_positions = pred_positions[path_indices].cpu().numpy()
                plt.plot(path_positions[:, 0], path_positions[:, 1])

            plt.show()

    def on_train_end(self, logs=None):
        pass  # No-op, but you can add final plotting here if needed
