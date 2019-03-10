import torch
from torch.utils.data import DataLoader
from data.CornellGrasp.dataset import build_dataset

def build_data_loader(cfg):
    cdset = build_dataset(cfg)
    dsl = DataLoader(cdset, batch_size=cfg.b_size, collate_fn=collate_fn)
    return dsl

def collate_fn(slist):
    """
    collate_fn. required to create a batch of examples from this dataset
    """

    x_img = torch.stack([e[0] for e in slist])
    y_bboxes = [e[1] for e in slist]
    return (x_img, y_bboxes)
