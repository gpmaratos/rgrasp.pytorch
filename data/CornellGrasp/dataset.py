import os
import torch
import skimage
import numpy as np
from torch.utils.data import Dataset
from utils.bbox import BoundingBoxList
from torchvision.transforms import Normalize

def build_dataset(cfg, train = True):
    return CornellDataset(cfg.d_path, train)

class CornellDataset(Dataset):
    """
    CornellDataset. Class which defines how a single image is processed,
    from the raw cornell grasping dataset. Images 132 and 165 are ignored
    because they have nans for some fields.

    Arguments:
        d_path (string): path to the folder containing all of the
            example images
    """

    def __init__(self, d_path, train):
        index = list(range(100, 950)) + list(range(1000, 1035))
        del index[32]; del index[64]
        index = np.array(index, dtype=int)
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]
        )

        self.train = train
        self.d_path = d_path
        self.index = index
        self.normalize = normalize

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ipref = os.path.join(self.d_path, "pcd%04d"%(self.index[idx]))
        ipath = ipref + "r.png"
        iarr = skimage.io.imread(ipath)
        with open(ipref+"cpos.txt") as f:
            f = f.read().split("\n")[:-1]
        bboxes = BoundingBoxList(f)
        #apply transformations here
        if self.train:
            iarr = torch.tensor(iarr).permute(2, 0, 1).float()
            iarr = self.normalize(iarr)
        return iarr, bboxes
