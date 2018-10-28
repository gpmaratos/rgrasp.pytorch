import os
import skimage
import numpy as np
from torch.utils.data import Dataset

class CornellDataset(Dataset):
    def __init__(self, dpath, iwidth, anum):
        """There were NaNs in image 132 and 165 so they are ignored.
        I assume the images are 640 X 480, and manually crop them
        with cXl through cYd"""

        self.index = np.array(list(range(100, 132))
            +list(range(133, 165)) + list(range(166, 950))
            + list(range(1000, 1035)), dtype=int)
        self.dpath = dpath
        self.iwidth = iwidth
        self.adim = iwidth // anum
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
            irecs = [self.get_tuple(irec, self.cXl, self.cYu)
                for irec in irecs]
        return torch.Tensor(iarr.transpose(2, 0, 1)).float(), irecs

    def get_coord(self, f):
        """given a string containing coordinates, convert them to
            a tuple of floats"""

        ln = f.split()
        return (float(ln[0]), float(ln[1]))

    def get_tuple(self, rec, wcrop, lcrop):
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
        return (x - wcrop, y - lcrop, width, height, ang)
