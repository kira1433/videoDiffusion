import h5py
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple

hf = h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/Originalface.hdf5", "r")
temp_videos = [] 
for temp in range(1000):
    video_no = '{:03}'.format(temp)
    frames = list(hf[f"Original/{video_no}"].keys())
    temp_list = []
    for frame in frames:
        temp_list.append(hf[f"Original/{video_no}/{frame}"][()])
        if(len(temp_list) == 108):
            break
    temp_videos.append(np.array(temp_list))
videos = np.array(temp_videos)
np.save("./data_108.npy", videos)