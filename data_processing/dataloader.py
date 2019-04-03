import torch
import numpy as np
from torch.utils.data import DataLoader
from data_processing.dataset import CornellDataset

def collate(slist):
    """collate_fn. make batches from dataset outputs"""

    tups = [item for lst in slist for item in lst]
    x_img = torch.stack([e[0] for e in tups])
    x_img_orig = np.stack([e[1] for e in tups])
    y_bboxes = [e[2] for e in tups]
    return (x_img, x_img_orig, y_bboxes)

def build_data_loader(batch_size, d_path, d_type, n_ang):
    """build_data_loader. create wrapper for dataset for batch processing"""

    ds = CornellDataset(d_path, d_type, n_ang)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate)

