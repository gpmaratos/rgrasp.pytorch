import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from utils.bbox import BoundingBoxList
from data.CornellGrasp.dataset import build_dataset

def build_visualizer(cfg):
    return Visualizer(cfg)

class Visualizer(Dataset):
    """
    Visualizer. Used to show the annotations on an image, and visualize
    the inference. It is a dataset object that gives images in a format
    for visualizing, and later, to perform inference call prepare()
    before passing into a network.
    """

    def __init__(self, dpath):
        index = list(range(100, 950)) + list(range(1000, 1035))
        del index[32]; del index[64]
        index = np.array(index, dtype=int)
        self.dpath = dpath
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ipref = os.path.join(self.dpath, "pcd%04d"%(self.index[idx]))
        ipath = ipref + "r.png"
        iarr = skimage.io.imread(ipath)
        with open(ipref+"cpos.txt") as f:
            f = f.read().split("\n")[:-1]
        bboxes = BoundingBoxList(f)
        return iarr, bboxes

    def show_ground_truth(self, iarr, bboxes):
        img = Image.fromarray(iarr)
        draw = ImageDraw.Draw(img)
        for rec in bboxes.irecs:
            draw.line(rec[:2], fill=(200, 0, 0))
            draw.line(rec[1:], fill=(0, 0, 0))
        plt.imshow(img)
        plt.show()

    def prepare(self):
        pass
