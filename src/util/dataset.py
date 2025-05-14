import torch
from torch.utils.data import Dataset
import numpy as np
import os


class DichasusDataset(Dataset):
    """
    PyTorch Dataset for preprocessed numpy arrays. Supports both labeled and unlabeled data.
    Can be initialized from a dict of numpy arrays or from a single .npz file path.
    """

    def __init__(self, data):
        """
        Args:
            data: dict of numpy arrays (with keys 'csi', 'time', and optionally 'pos'),
                  or str (path to .npz file)
        """
        if isinstance(data, str):
            # Assume it's a .npz file name
            root_dir = os.path.dirname(os.path.abspath(__file__))
            npz_path = os.path.join(root_dir, "../../data", data)
            loaded = np.load(npz_path)
            self.csi = loaded['csi']
            self.time = loaded['time']
            self.pos = loaded['pos'] if 'pos' in loaded.files else None
        elif isinstance(data, dict):
            self.csi = data['csi']
            self.time = data['time']
            self.pos = data.get('pos', None)
        else:
            raise ValueError(
                "DichasusDataset expects a dict of numpy arrays or a .npz file path.")
        self.labeled = self.pos is not None

    def __len__(self):
        return len(self.csi)

    def __getitem__(self, idx):
        csi = torch.from_numpy(self.csi[idx])
        time = torch.from_numpy(np.array(self.time[idx]))
        if self.labeled:
            pos = torch.from_numpy(self.pos[idx])
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
    out_path = os.path.join(root_dir, "../../data", out_file)
    np.savez_compressed(out_path, **dict)
    return out_path
