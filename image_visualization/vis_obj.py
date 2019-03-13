import os
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def show_obj(inp):
    """
    show_obj. Function that displays images of an object in the dataset
    """
    if len(inp) == 1:
        plt.imshow(inp[0])
    if len(inp) == 2:
        _, axs = plt.subplots(2, 1)
        axs[0].imshow(inp[0])
        axs[1].imshow(inp[1])
    if len(inp) == 3:
        _, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(inp[0])
        axs[0, 1].imshow(inp[1])
        axs[1, 0].imshow(inp[2])
    if len(inp) == 4:
        _, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(inp[0])
        axs[0, 1].imshow(inp[1])
        axs[1, 0].imshow(inp[2])
        axs[1, 1].imshow(inp[3])
    plt.show()
