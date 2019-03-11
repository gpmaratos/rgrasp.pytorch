import os
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

#I should split these objects in the future so that I don't import matplotlib

class CGObject(Dataset):
    """
    CGObject. Class which iterates over objects in the Cornell Grasp Dataset
    The index is a list of numpy arrays. The numpy arrays are views of
    an array that has all the indices for images.

    Arguments:
        d_path (string): path to the folder containing all of the
            example images
    """

    def build_index(self, index):
        index_list = []
        index_list.append(index[:2])
        for i in range(2, len(index), 4):
            index_list.append(index[i:i+4])
        return index_list

    def __init__(self, d_path):
        index = list(range(100, 950)) + list(range(1000, 1035))
        del index[32]; del index[64]
        index = np.array(index, dtype=int)
        index = build_index(index)
        self.d_path = d_path
        self.index = index

    def __len__(self):
        return len(self.index_list)

    def get_img(self, idx):
        ipref = os.path.join(self.d_path, 'pcd%04d'%(idx))
        ipath = ipref + 'r.png'
        iarr = imread(ipath)
        return iarr

    def __getitem__(self, idx):
        indices = self.index_list[idx]
        imgs = [self.get_img(i) for i in indices]
        return imgs

def show_obj(inp):
    """
    show_obj. Function that displays images of an object in the dataset
    """
    img = [Image.fromarray(i) for i in inp]
    if len(img) == 2:
        fig = plt.figure(figsize=(2, 1))
        fig.add_subplot(
    plt.imshow(img)
    plt.show()
