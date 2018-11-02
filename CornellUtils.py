import os
import math
import skimage
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class CornellDataset(Dataset):
    def __init__(self, dpath, aug, k):
        """There were NaNs in image 132 and 165 so they are ignored.
        FIXED: output image dimensions, and dimension of the feature map
        I assume the images are 640 X 480, and manually crop them
        with cXl through cYd"""

        self.dpath = dpath
        self.aug = aug
        self.index = np.array(list(range(100, 132))
            +list(range(133, 165)) + list(range(166, 950))
            + list(range(1000, 1035)), dtype=int)
        self.angs = np.array([math.radians(i) for i in range(90, -90, -180//k)])
        self.iwidth = 320
        self.anum = 10
        self.adim = self.iwidth // self.anum
        self.cXl = 120
        self.cXr = 200
        self.cYu = 100
        self.cYd = 60

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """Images are loaded on demand, currently it returns
            The image as a Torch tensor and the tuple representing
            a rectangle"""

        ipref = os.path.join(self.dpath, "pcd%04d"%(self.index[idx]))
        ipath = ipref + "r.png"
        iarr = skimage.io.imread(ipath)
        iarr = skimage.util.crop(iarr, ((self.cYu, self.cYd),
            (self.cXl, self.cXr), (0, 0)))
        with open(ipref+"cpos.txt") as f:
            f = f.read().split("\n")[:-1]
            irecs = [[self.get_coord(f[i]), self.get_coord(f[i+1]),
                        self.get_coord(f[i+2]), self.get_coord(f[i+3])]
                            for i in range(0, len(f), 4)]
        irecs = [[(x-self.cXl, y-self.cYu) for x, y in rec]
            for rec in irecs]
        #IMAGE TRANSFORMATIONS
        import pdb;pdb.set_trace()
        if self.aug:
            theta = round(random.uniform(-30, 30), 3)
            iarr = skimage.transform.rotate(iarr, theta,
            theta = math.radians(-1*theta)
                preserve_range=True)
            if random.randint(0, 1):
                iarr = np.flipud(iarr)
                fhorz = True
            else:
                iarr = np.flipud(iarr)
                fhorz = False
            irecs = [[(coord[0] - 160, coord[1] - 160)
                for coord in coords] for coords in irecs]
            ct = math.cos(theta)
            st = math.sin(theta)
            irecs = [[(ct*coord[0] - st*coord[1],
                        st*coord[0] + ct*coord[1])
                        for coord in coords] for coords in irecs]
            irecs = [[(coord[0] + 160, coord[1] + 160)
            if fhorz:
                irecs = [[(320 - coord[0], coord[1]) for coord in coords]
                    for coords in rrecs]
            else:
                irecs = [[coord[0], 320 - coord[1]) for coord in coords]
                    for coords in rrecs]
        import pdb;pdb.set_trace()
        irecs = [self.get_tuple(irec) for irec in irecs]
        return torch.Tensor(iarr.transpose(2, 0, 1)).float(), irecs

    def get_coord(self, f):
        """given a string containing coordinates, convert them to
            a tuple of floats"""

        ln = f.split()
        return (float(ln[0]), float(ln[1]))

    def get_tuple(self, rec):
        """given a set of coordinates representing a rectangle,
            compute the tuple t"""

        xhat = rec[0][0] - rec[-1][0]
        yhat = rec[0][1] - rec[-1][1]
        width = math.sqrt(xhat**2 + yhat**2)
        xhat = rec[0][0] - rec[1][0]
        yhat = rec[0][1] - rec[1][1]
        height = math.sqrt(xhat**2 + yhat**2)
        x = float(sum([point[0] for point in rec]))/4
        y = float(sum([point[1] for point in rec]))/4
        if rec[0][0] < rec[1][0]:
            xhat = rec[1][0] - rec[0][0]
            yhat = rec[1][1] - rec[0][1]
            ang = math.atan2(-1*yhat, xhat)
        else:
            xhat = rec[0][0] - rec[1][0]
            yhat = rec[0][1] - rec[1][1]
            ang = math.atan2(-1*yhat, xhat)
        xpos = x // self.adim
        ypos = y // self.adim
        apos = self.get_anchor_ind(ang)
        return ((x - (xpos*self.adim + self.adim/2)) / self.adim,
                (y - (ypos*self.adim + self.adim/2)) / self.adim,
                math.log(width / self.adim),
                math.log(height / self.adim),
                (ang - self.angs[apos])/(180/len(self.angs)),
                xpos, ypos, apos)

    def get_anchor_ind(self, angle):
        """retrieve the index of the closest anchor"""
        return np.abs(self.angs-angle).argmin()
