import torch
from torch.utils.data import DataLoader
from data_processing.dataset import build_dataset

def build_data_loader(cfg):
    cdset = build_dataset(cfg, aug=False)
    dsl = DataLoader(cdset, batch_size=cfg.b_size, collate_fn=collate_fn)
    return dsl

def collate_fn(slist):
    """
    collate_fn. required to create a batch of examples from this dataset
    """

    tups = [item for lst in slist for item in lst]
    x_img = torch.stack([e[0] for e in tups])
    y_bboxes = [e[1] for e in tups]
    return (x_img, y_bboxes)
