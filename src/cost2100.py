"""Canonical COST2100 reader for the public CsiNet/CRNet/TransNet files."""

from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class COST2100Dataset(Dataset):
    """Load one official MAT split as float32 [N, 2, 32, 32].

    The two channels preserve the public implementations' convention:
    channel 0 is real, channel 1 is imaginary. The values remain in [0, 1];
    subtract 0.5 only when constructing complex values for metrics.
    """

    def __init__(self, mat_path: str | Path, variable_name: str = "HT") -> None:
        mat_path = Path(mat_path)
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)

        loaded = scipy.io.loadmat(mat_path, variable_names=[variable_name])
        if variable_name not in loaded:
            raise KeyError(f"{variable_name!r} not found in {mat_path}")

        data = np.asarray(loaded[variable_name])
        if data.ndim != 2 or data.shape[1] != 2048:
            raise ValueError(
                f"Expected {variable_name} with shape [N, 2048], got {data.shape}"
            )
        if not np.issubdtype(data.dtype, np.number):
            raise TypeError(f"Expected numeric COST2100 data, got {data.dtype}")

        data = np.asarray(data, dtype=np.float32, order="C")
        self.data = torch.from_numpy(data.reshape(data.shape[0], 2, 32, 32))
        self.source_path = mat_path

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]
