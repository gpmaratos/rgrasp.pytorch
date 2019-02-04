import os
import math
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

    def draw_tuple(self, draw, tup):
#        import pdb;pdb.set_trace()
        draw.ellipse((tup[0]-3, tup[1]-3, tup[0]+3, tup[1]+3),
            fill=(0, 0, 200))
        alpha = math.cos(math.radians(90)-tup[2])*10
        beta = math.sin(math.radians(90)-tup[2])*10
        x_ray = tup[0]+alpha
        y_ray = tup[1]+beta
        draw.line((tup[0], tup[1], x_ray, y_ray),
            fill=(0, 200, 0))

    def show_ground_truth(self, iarr, bboxes):
        img = Image.fromarray(iarr)
        draw = ImageDraw.Draw(img)
        for i in range(len(bboxes.ibboxes)):
            rec = bboxes.irecs[i]
            tup = bboxes.ibboxes[i]
            draw.line(rec[:2], fill=(200, 0, 0))
            draw.line(rec[1:], fill=(0, 0, 0))
            self.draw_tuple(draw, tup)
        plt.imshow(img)
        plt.show()

    def prepare(self):
        pass
