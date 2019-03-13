import os
import torch
import skimage
import random
import numpy as np
from torch.utils.data import Dataset
from structures.bbox import BoundingBoxList
from torchvision.transforms import Normalize
from data_processing.augment_image import build_augmenter

def build_dataset(cfg, train = True, aug = True, dset = 'train'):
    return CornellDataset(cfg, train, aug, dset)

class CornellDataset(Dataset):
    """
    CornellDataset. Class which defines how a single image is processed,
    from the raw cornell grasping dataset. Images 132 and 165 are ignored
    because they have nans for some fields.

    Arguments:
        d_path (string): path to the folder containing all of the
            example images
    """

    def __init__(self, cfg, train, aug, dset):
        #first extract parameters from config and build image normalizer
        d_path = cfg.d_path
        if aug:
            x_dim = cfg.x_dim
            y_dim = cfg.y_dim
            augment = build_augmenter(x_dim, y_dim)
            self.augment = augment
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]
        )

        #then create object lookup table and background mapping
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

        #then create train test split
        random.seed(4211)
        obj_ids = list(lookup_dict.keys())
        random.shuffle(obj_ids)
        split = int(0.8*len(obj_ids))
        if dset == 'train':
            index = np.array(obj_ids[:split])
        if dset == 'test':
            index = np.array(obj_ids[split:])

        #finally create class members
        self.train = train
        self.d_path = d_path
        self.normalize = normalize
        self.aug = aug
        self.lookup_dict = lookup_dict
        self.bkg_dict = bkg_dict
        self.index = index

    def __len__(self):
        return len(self.index)

    def get_img(self, idx):
        ipref = os.path.join(self.d_path, "pcd%04d"%(idx))
        ipath = ipref + "r.png"
        iback = self.bkg_dict[idx]
        bpath = os.path.join(self.d_path, "pcdb%04d"%(iback)+"r.png")
        iarr = skimage.io.imread(ipath)
        iarr -= skimage.io.imread(bpath)
        with open(ipref+"cpos.txt") as f:
            f = f.read().split("\n")[:-1]
        bboxes = BoundingBoxList(f)
        if self.aug:
            iarr, bboxes = self.augment(iarr, bboxes)
        if self.train:
            iarr = torch.tensor(iarr).permute(2, 0, 1).float()
            iarr = self.normalize(iarr)
        return iarr, bboxes

    def __getitem__(self, idx):
        obj_index = self.index[idx]
        indices = self.lookup_dict[obj_index]
        imgs = [self.get_img(i) for i in indices]
        return imgs
