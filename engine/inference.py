import numpy as np
from inference.infer import Infer
from torchvision.transforms import Normalize
from data.CornellGrasp.dataset import build_dataset

def infer(model, cfg, data = None):
    model.eval()
    inf = Infer()
    if data == None:
        ds = build_dataset(cfg, train=False)
        for inds, data in enumerate(ds):
            pred = inf(model, data[0])
