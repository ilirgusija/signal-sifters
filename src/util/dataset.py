import torch
from torch.utils.data import Dataset
import numpy as np
import os
from src.models.gaussian_dissimilarity import GaussianDissimilarityModel
from tqdm.auto import tqdm
import multiprocessing as mp

class DichasusDataset(Dataset):
    """
    PyTorch Dataset for preprocessed numpy arrays. Supports both labeled and unlabeled data.
    Can be initialized from a dict of numpy arrays or from a single .npz file path.
    """

    def __init__(self, data, device='cpu'):
        """
        Args:
            data: dict of numpy arrays (with keys 'csi', 'time', and optionally 'pos'),
                  or str (path to .npz file)
        """
        if isinstance(data, str):
            # Assume it's a .npz file name
            root_dir = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(root_dir, "../../data/processed/", data)
            loaded = np.load(npz_path)
            self.csi = torch.from_numpy(loaded['csi'])
            self.time = torch.from_numpy(loaded['time'])
            self.pos = torch.from_numpy(
                loaded['pos']) if 'pos' in loaded.files else None
        elif isinstance(data, dict):
            self.csi = torch.from_numpy(data['csi'])
            self.time = torch.from_numpy(data['time'])
            pos_data = data.get('pos', None)
            self.pos = torch.from_numpy(
                pos_data) if pos_data is not None else None
        else:
            raise ValueError(
                "DichasusDataset expects a dict of numpy arrays or a .npz file path.")
        self.csi = self.csi.to(device)
        self.time = self.time.to(device)
        self.labeled = self.pos is not None
        if self.labeled:
            # MPS does not support float64, so we convert to float32 before moving to device
            if device == torch.device("mps"):
                self.pos = self.pos.to(torch.float32)
            self.pos = self.pos.to(device)

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, idx):
        csi = self.csi[idx]
        time = self.time[idx]
        if self.labeled:
            pos = self.pos[idx]
            return csi, pos, time
        else:
            return csi, time


def save_numpy_dataset(dict, out_file="dataset.npz"):
    """
    Utility to save numpy arrays to a single .npz file for later loading.
    Args:
        csi: np.ndarray
        time: np.ndarray
        pos: np.ndarray or None
        out_path: str, path to save the .npz file
    Returns:
        out_path (str)
    """
    # Get root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(root_dir, "../../data/processed", out_file)
    np.savez_compressed(out_path, **dict)
    return out_path


class ChannelChartingPathDataset(Dataset):
    """
    PyTorch Dataset for generating random path batches for ChannelChart training.
    Each item is a tuple (indices, y_true) for a batch.
    """
    
    def __init__(self, GDM: GaussianDissimilarityModel, csi_time_domain, training_batches=2000, max_pathhops=100,
                 min_batch_size=1500, max_batch_size=4000, min_hoplength=0.25, max_hoplength=10.0,
                 randomize_pathhops=False, device='cpu'):
        self.GDM = GDM
        self.csi_time_domain = csi_time_domain
        self.training_batches = training_batches
        self.max_pathhops = max_pathhops
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_hoplength = min_hoplength
        self.max_hoplength = max_hoplength
        self.randomize_pathhops = randomize_pathhops
        self.device = device

        self.y_true_pregenerated = dict()
        with tqdm(total=training_batches + 5, desc="Pre-computing training paths") as pbar:
            todo_queue = mp.Queue()
            output_queue = mp.Queue()

            for batch in tqdm(range(training_batches + 5), desc="Preparing multiprocessing inputs"):
                todo_queue.put(batch)

            process_count = mp.cpu_count()

            for _ in tqdm(range(process_count), desc="Starting Processes"):
                todo_queue.put(-1)
                p = mp.Process(target=self.batch_generator_worker,
                               args=(todo_queue, output_queue))
                p.start()

            finished_processes = 0
            while finished_processes != process_count:
                batch_count, y_true = output_queue.get()

                if batch_count == -1:
                    finished_processes = finished_processes + 1
                else:
                    self.y_true_pregenerated[batch_count] = y_true
                    pbar.update(1)
                    
    def batch_generator_worker(self, todo_queue, output_queue):
        while True:
            batch_count = todo_queue.get()

            if batch_count == -1:
                output_queue.put((-1, None))
                break

            # Determine current batch size
            batch_size = int(
                round(
                    batch_count / self.training_batches *
                    (self.max_batch_size - self.min_batch_size) + self.min_batch_size
                )
            )

            # Round batch size to nearest steps of 200 to prevent re-tracing compute graph too often
            batch_size = int(round(batch_size / 200) * 200)

            # Determine number of hops for current subsampling ratio
            pathhops_length_limit = (batch_count / self.training_batches) ** 0.15 * (
                self.min_hoplength - self.max_hoplength) + self.max_hoplength
            if self.randomize_pathhops:
                pathhops_maxlength = torch.rand(
                    size=(batch_size,), device=self.device) * (self.max_hoplength - self.pathhops_length_limit) + pathhops_length_limit
            else:
                pathhops_maxlength = torch.ones(
                    batch_size, dtype=torch.float32) * pathhops_length_limit

            # Generate random short paths and assemble y_true, consisting of batch_size paths, each made up of
            # * number of path hops
            # * mean value of dissimilarity random variable
            # * variance of  dissimilarity random variable
            # * datapoint indices along path; ends with repeating last index if too few hops
            paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances = self.GDM.get_random_short_paths(batch_size, pathhops_maxlength)
            assert torch.all(path_hops <= paths.shape[1] - 1), (
                f"Found path_hops > max index: max(path_hops)={path_hops.max()} vs paths.shape[1]-1={paths.shape[1]-1}"
            )
            paths = paths[:, :self.max_pathhops + 1]
            y_true = torch.hstack([
                # None is analogous to np.newaxis
                path_hops[:, None],
                total_dissimilarity_means[:, None],
                total_dissimilarity_variances[:, None],
                paths
            ]).to(self.device)
            output_queue.put((batch_count, y_true))

    # def __init__(self, GDM: GaussianDissimilarityModel, csi_time_domain, training_batches=2000, max_pathhops=100,
    #              min_batch_size=1500, max_batch_size=4000, min_hoplength=0.25, max_hoplength=10.0,
    #              randomize_pathhops=False, device='cpu'):
    #     print("Initializing ChannelChartingPathDataset")
    #     self.GDM = GDM
    #     self.csi_time_domain = csi_time_domain
    #     self.training_batches = training_batches
    #     self.max_pathhops = max_pathhops
    #     self.min_batch_size = min_batch_size
    #     self.max_batch_size = max_batch_size
    #     self.min_hoplength = min_hoplength
    #     self.max_hoplength = max_hoplength
    #     self.randomize_pathhops = randomize_pathhops
    #     self.device = device

    #     # Pregenerate all batches and store them in a list
    #     self.pregenerated_batches = []
    #     indices = torch.arange(
    #         self.csi_time_domain.shape[0], device=self.device)
    #     for batch_count in tqdm(range(self.training_batches), desc="Pre-computing training paths"):

    #         # Determine current batch size
    #         batch_size = int(
    #             round(
    #                 batch_count / self.training_batches *
    #                 (self.max_batch_size - self.min_batch_size) + self.min_batch_size
    #             )
    #         )
    #         # Round batch size to nearest steps of 200 to prevent re-tracing compute graph too often
    #         batch_size = int(round(batch_size / 200) * 200)

    #         # Determine number of hops for current subsampling ratio
    #         pathhops_length_limit = (batch_count / self.training_batches) ** 0.15 * (
    #             self.min_hoplength - self.max_hoplength) + self.max_hoplength
    #         if self.randomize_pathhops:
    #             pathhops_maxlength = np.random.uniform(
    #                 pathhops_length_limit, self.max_hoplength, size=batch_size)
    #         else:
    #             pathhops_maxlength = np.ones(
    #                 batch_size, dtype=np.float32) * pathhops_length_limit

    #         # Generate random short paths and assemble y_true, consisting of batch_size paths, each made up of
    #         # * number of path hops
    #         # * mean value of dissimilarity random variable
    #         # * variance of  dissimilarity random variable
    #         # * datapoint indices along path; ends with repeating last index if too few hops
    #         paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances = self.GDM.get_random_short_paths(
    #             batch_size, pathhops_maxlength)
    #         # print("Assertion check:")
    #         assert torch.all(path_hops <= paths.shape[1] - 1), (
    #             f"Found path_hops > max index: max(path_hops)={path_hops.max()} vs paths.shape[1]-1={paths.shape[1]-1}"
    #         )
    #         paths = paths[:, :self.max_pathhops + 1]
    #         y_true = torch.hstack([
    #             # None is analogous to np.newaxis
    #             path_hops[:, None],
    #             total_dissimilarity_means[:, None],
    #             total_dissimilarity_variances[:, None],
    #             paths
    #         ]).to(self.device)
    #         y_true = y_true.to(self.device)
    #         self.pregenerated_batches.append((indices, y_true))

    def save(self):
        # Save pregenerated_batches and any other relevant info
        data_dir = os.path.join(os.getcwd(), "../data/processed/preprocessed")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "y_true_pregenerated.pt")
        print(f"Saving y_true_pregenerated to {path}")
        data = {
            'y_true_pregenerated': self.y_true_pregenerated,
            'max_pathhops': self.max_pathhops,
            'training_batches': self.training_batches,
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'min_hoplength': self.min_hoplength,
            'max_hoplength': self.max_hoplength,
            'randomize_pathhops': self.randomize_pathhops,
            # Add any other attributes you want to restore
        }
        torch.save(data, path)

    @classmethod
    def load(cls, GDM, csi_time_domain, device='cpu'):
        # Load pregenerated_batches and create a new instance
        obj = cls.__new__(cls)  # Create instance without calling __init__
        data_dir = os.path.join(os.getcwd(), "../data/processed/preprocessed")
        path = os.path.join(data_dir, "y_true_pregenerated.pt")
        data = torch.load(path, map_location=device)
        # Manually set attributes
        obj.GDM = GDM
        obj.csi_time_domain = csi_time_domain
        obj.device = device
        obj.max_pathhops = data['max_pathhops']
        obj.training_batches = data['training_batches']
        obj.y_true_pregenerated = data['y_true_pregenerated']
        obj.min_batch_size = data['min_batch_size']
        obj.max_batch_size = data['max_batch_size']
        obj.min_hoplength = data['min_hoplength']
        obj.max_hoplength = data['max_hoplength']
        obj.randomize_pathhops = data['randomize_pathhops']
        # Set any other attributes as needed
        return obj

    def __len__(self):
        return self.training_batches

    def __getitem__(self, batch_count):
        return self.pregenerated_batches[batch_count]
