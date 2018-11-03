import torch
import math
import numpy as np
from CornellUtils import CornellDataset
from GraspNetwork import graspn
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

#DELETE WHEN DONE WITH DEMO FUNCTION
import skimage
import os

class Display(CornellDataset):
    def __init__(self, dpath, bbone, wpath, numa, aug):
        super().__init__(dpath, aug, numa)
        self.net = graspn(bbone, numa)
        self.net.load_state_dict(torch.load(wpath, map_location="cpu"))
        self.net.eval()

    def display(self, idx):
        dataX, dataY = self.__getitem__(idx)
        img = dataX.permute(1, 2, 0).numpy().astype(np.uint8)
        img = Image.fromarray(img, "RGB")
        draw = ImageDraw.Draw(img)
        dataX = dataX.reshape(-1, 3, self.iwidth, self.iwidth)
        preg, pcls = self.net(dataX)
        preds = pcls[:, 1] - pcls[:, 0]
        _, inds = torch.sort(preds , descending=True)
        for ind in inds[:5]:
            t = preg[ind]
            #need to translate ind to 3d
            a = ind // (self.anum**2)
            x = (ind // self.anum) % self.anum
            y = ind % self.anum

            x = x*self.adim + self.adim/2
            y = y*self.adim + self.adim/2

            px = x.item() + (t[0]*self.adim).item()
            py = y.item() + (t[1]*self.adim).item()
            pw = (torch.exp(t[2])*self.adim).item()
            pl = (torch.exp(t[3])*self.adim).item()
            pt = (t[4]*(180/len(self.angs)) + self.angs[a]).item()

            trans = torch.Tensor([[math.cos(pt), -1*math.sin(pt)],\
                                    [math.sin(pt), math.cos(pt)]])
            roff = torch.Tensor([-1*pl/2, pw/2])
            tl = torch.matmul(trans, roff) + torch.Tensor([px, py])
            roff = torch.Tensor([pl/2, pw/2])
            tr = torch.matmul(trans, roff) + torch.Tensor([px, py])
            roff = torch.Tensor([pl/2, -1*pw/2])
            br = torch.matmul(trans, roff) + torch.Tensor([px, py])
            roff = torch.Tensor([-1*pl/2, -1*pw/2])
            bl = torch.matmul(trans, roff) + torch.Tensor([px, py])
            coords= [tuple(tl.tolist()), tuple(tr.tolist()),\
                    tuple(br.tolist()), tuple(bl.tolist())]

            draw.line(coords, fill=(200, 0, 0))
        plt.imshow(img)
        plt.show()

    def demo(self, idx):
        """demo function that will display images and their
            transformations """
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
        with open(ipref+"cpos.txt") as f:
            f = f.read().split("\n")[:-1]
            irecs = [[self.get_coord(f[i]), self.get_coord(f[i+1]),
                        self.get_coord(f[i+2]), self.get_coord(f[i+3])]
                            for i in range(0, len(f), 4)]
            modirec = [[(x-self.cXl, y-self.cYu) for x, y in rec] for rec in irecs]

        #DISPLAY UNMODIFIED IMAGE
        img = Image.fromarray(iarr, "RGB")
        draw = ImageDraw.Draw(img)
        [(draw.line(rec[:2], fill=(200, 0, 0))\
            ,draw.line(rec[1:], fill=(0, 0, 0))) for rec in modirec]
        plt.imshow(img)
        plt.show()

        #ROTATE IMAGE
        theta = -45
        rimg = skimage.transform.rotate(iarr, theta, preserve_range=True)
        rrecs = [[(coord[0] - 160, coord[1] - 160) for coord in coords] for coords in modirec]
        theta = math.radians(-1*theta)
        rrecs = [[(math.cos(theta)*coord[0] - math.sin(theta)*coord[1],
                    math.sin(theta)*coord[0] + math.cos(theta)*coord[1])
                    for coord in coords] for coords in rrecs]
        rrecs = [[(coord[0] + 160, coord[1] + 160) for coord in coords] for coords in rrecs]
        rimg = Image.fromarray(rimg.astype('uint8'), "RGB")
        draw = ImageDraw.Draw(rimg)
        [(draw.line(rec[:2], fill=(200, 0, 0))\
            ,draw.line(rec[1:], fill=(0, 0, 0))) for rec in rrecs]
        plt.imshow(rimg)
        plt.show()

        #FLIP THE IMAGE LR AFTER ROTATION
        theta = -45
        rfimg = skimage.transform.rotate(iarr, theta, preserve_range=True)
        rfimg = np.fliplr(rfimg)
        rrecs = [[(coord[0] - 160, coord[1] - 160) for coord in coords] for coords in modirec]
        theta = math.radians(-1*theta)
        rrecs = [[(math.cos(theta)*coord[0] - math.sin(theta)*coord[1],
                    math.sin(theta)*coord[0] + math.cos(theta)*coord[1])
                    for coord in coords] for coords in rrecs]
        rrecs = [[(coord[0] + 160, coord[1] + 160) for coord in coords] for coords in rrecs]
        rrecs = [[(320 - coord[0], coord[1]) for coord in coords] for coords in rrecs]
        rfimg = Image.fromarray(rfimg.astype('uint8'), "RGB")
        draw = ImageDraw.Draw(rfimg)
        [(draw.line(rec[:2], fill=(200, 0, 0))\
            ,draw.line(rec[1:], fill=(0, 0, 0))) for rec in rrecs]
        plt.imshow(rfimg)
        plt.show()

        #FLIP THE IMAGE UD AFTER ROTATION
        theta = -45
        rfimg = skimage.transform.rotate(iarr, theta, preserve_range=True)
        rfimg = np.flipud(rfimg)
        rrecs = [[(coord[0] - 160, coord[1] - 160) for coord in coords] for coords in modirec]
        theta = math.radians(-1*theta)
        rrecs = [[(math.cos(theta)*coord[0] - math.sin(theta)*coord[1],
                    math.sin(theta)*coord[0] + math.cos(theta)*coord[1])
                    for coord in coords] for coords in rrecs]
        rrecs = [[(coord[0] + 160, coord[1] + 160) for coord in coords] for coords in rrecs]
        rrecs = [[(coord[0], 320 - coord[1]) for coord in coords] for coords in rrecs]
        rfimg = Image.fromarray(rfimg.astype('uint8'), "RGB")
        draw = ImageDraw.Draw(rfimg)
        [(draw.line(rec[:2], fill=(200, 0, 0))\
            ,draw.line(rec[1:], fill=(0, 0, 0))) for rec in rrecs]
        plt.imshow(rfimg)
        plt.show()
