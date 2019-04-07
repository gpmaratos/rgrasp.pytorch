import os
import math
import skimage
import random
import torch
import numpy as np
from target_formatting.five_tuple import ConvertTuple
from data_processing.augment_image import Augmenter
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

class CornellDataset(Dataset):
    """
    CornellDataset. Class which defines how one example is processed from the
    raw data. The image is subtracted from its background, then normalzed based
    on the requirements specified in the pytorch documentation, and hopefully
    (future feature) be cropped and rotated randomly. Images 132 and 165 are
    ignored because their ground truth is not annotated properly.

    Applied image cropping so that images are 320 320 crops of the center

    Arguments:
        d_path (string): path to the folder containing all the data. It needs
            to contain all the images of objects and the background, and z.txt
            which is used to do the object wise split.

        d_type (string): accepted values are 'train', 'val', and 'test'. This
            specifies which part of the dataset to retrieve. It is split, object
            wise, 70/10/20 train/val/test using a random seed which should
            remain fixed for all experiments to stay consistent. In the future
            I hope to implement a LOOCV method for better evaluation.
    """

    def __init__(self, d_path, d_type):
        #build objects for data augmentation
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]
        )
        img_size = (320, 320)
        augmenter = Augmenter(img_size[0], img_size[1])
        rec_formatter = ConvertTuple()

        #build object_lookup_table and background_lookup_table
        with open(os.path.join(d_path, 'z.txt')) as f:
            data = f.read().split('\n')[:-1]
        obj_lt, bkg_lt = {}, {}
        for line in data:
            p_line = line.split()
            img_id, obj_num = int(p_line[0]), int(p_line[1])
            background_img = int(p_line[3].split('_')[1])
            if img_id >= 100 and img_id < 950 or\
                img_id >= 1000 and img_id < 1035:
                if not img_id in [132, 165]:
                    if obj_num in obj_lt:
                        if not img_id in obj_lt[obj_num]:
                            obj_lt[obj_num].append(img_id)
                            bkg_lt[img_id] = background_img
                    else:
                        obj_lt[obj_num] = [img_id]
                        bkg_lt[img_id] = background_img

        #select dataset partition
        random.seed(4211)
        obj_ids = list(obj_lt.keys())
        random.shuffle(obj_ids)
        split_factor = int(0.1*len(obj_ids))
        if d_type == 'train':
            index = np.array(obj_ids[:7*split_factor])
        if d_type == 'val':
            index = np.array(obj_ids[7*split_factor:8*split_factor])
        if d_type == 'test':
            index = np.array(obj_ids[8*split_factor:])

        #create class members
        self.d_path = d_path
        self.normalize = normalize
        self.augmenter = augmenter
        self.obj_lt = obj_lt
        self.bkg_lt = bkg_lt
        self.index = index
        self.rec_formatter = rec_formatter

    def __len__(self):
        return len(self.index)

    def get_coord(self, f):
        #image cropping offsets are subtracted here
        ln = f.split()
        return (float(ln[0]) - 120, float(ln[1]) - 100)

    def extract_img(self, img_id):
        """extract_img. Function that extracts a single image using its id"""

        #read image
        img_pref = os.path.join(self.d_path, "pcd%04d"%(img_id))
        img_path = img_pref + "r.png"
        np_img = skimage.io.imread(img_path)

        #subtract background image
        background = self.bkg_lt[img_id]
        bkg_path = os.path.join(self.d_path, "pcdb%04d"%(background)+"r.png")
        np_img -= skimage.io.imread(bkg_path)

        #crop image to 320 by 320
        np_img = skimage.util.crop(np_img, ( (100, 60), (120, 200), (0, 0)))

        #extract ground truth rectangles
        with open(img_pref+"cpos.txt") as f:
            f = f.read().split("\n")[:-1]
        gt_recs = [
            [self.get_coord(f[i]),
            self.get_coord(f[i+1]),
            self.get_coord(f[i+2]),
            self.get_coord(f[i+3])]
                for i in range(0, len(f), 4)
        ]

        #and augment image for training
        np_img_mod, gt_recs_mod = self.augmenter(np_img, gt_recs)
        gt_recs_mod = [self.rec_formatter(rec) for rec in gt_recs_mod]
        np_img_mod = torch.tensor(np_img_mod).permute(2, 0, 1).float()
        np_img_mod = self.normalize(np_img_mod)

        return np_img_mod, np_img, gt_recs_mod, gt_recs

    def __getitem__(self, idx):
        obj_id = self.index[idx]
        img_ids = self.obj_lt[obj_id]
        img_gt_pairs = [self.extract_img(img_id) for img_id in img_ids]
        return img_gt_pairs

