import numpy as np
from utils.infer import Infer
from data.CornellGrasp.dataset import build_dataset

# Passing no data to inference means that inference is done with training
# data

def infer(model, cfg, data = None):
    model.eval()
    inf = Infer(cfg, model)
    if data == None:
        ds = build_dataset(cfg, train=False)
        for inds, data in enumerate(ds):
            pred = inf(model, data[0])
