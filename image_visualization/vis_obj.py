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

    def build_index(self, d_path):
        """
        build_index. z.txt contains the object ids for the images
        in the dataset. parse this file to determine what will be
        the indexible elements of this dataset.
        """

        with open(os.path.join(d_path, 'z.txt')) as f:
            data = f.read().split('\n')[:-1]
        lookup_dict = {}
        bkg_dict = {}
        for line in data:
            p_line = line.split()
            image_id, obj_num = int(p_line[0]), int(p_line[1])
            background_img = int(p_line[3].split('_')[1])
            if image_id >= 100 and image_id < 950 or\
                image_id >= 1000 and image_id < 1035:
                if not image_id in [135, 165]:
                    if obj_num in lookup_dict:
                        if not image_id in lookup_dict[obj_num]:
                            lookup_dict[obj_num].append(image_id)
                            bkg_dict[image_id] = background_img
                    else:
                        lookup_dict[obj_num] = [image_id]
                        bkg_dict[image_id] = background_img
        return lookup_dict, bkg_dict

    def __init__(self, d_path):
        lookup_dict, bkg_dict = self.build_index(d_path)
        self.d_path = d_path
        self.lookup_dict = lookup_dict
        self.index = np.array(list(lookup_dict.keys()))
        self.bkg_dict = bkg_dict

    def __len__(self):
        return len(self.index)

    def get_img(self, idx):
        ipref = os.path.join(self.d_path, 'pcd%04d'%(idx))
        ipath = ipref + 'r.png'
        iarr = imread(ipath)
        return iarr

    def __getitem__(self, idx):
        obj_index = self.index[idx]
        indices = self.lookup_dict[obj_index]
        imgs = [self.get_img(i) for i in indices]
        return imgs

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
