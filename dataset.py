import h5py
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple

class CustomVideoDataset(Dataset):
    def __init__(self, npy_path: str):
        super(CustomVideoDataset, self).__init__()
        self.videos = np.load(npy_path)
        pool = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.videos = pool(torch.tensor(self.videos)).numpy()
        self.videos = self.videos.transpose(0, 2, 1, 3, 4)
        self.videos = self.videos[:, :, :8, :, :]

    def __len__(self) -> int:
        return self.videos.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.videos[idx]

data = CustomVideoDataset("./data_108.npy")
print(data.__getitem__(0).shape)
print(type(data.__getitem__(0)))

