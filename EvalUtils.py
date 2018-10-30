import torch
import math
import numpy as np
from CornellUtils import CornellDataset
from GraspNetwork import graspn
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw

class Display(CornellDataset):
    def __init__(self, dpath, bbone, wpath, numa):
        super().__init__(dpath, numa)
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
        for ind in inds[:2]:
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
