import torch
from CornellUtils import CornellDataset
from GraspNetwork import graspn

class Display(CornellDataset):
    def __init__(self, dpath, bbone, wpath, numa):
        super(dpath, numa).__init__()
        self.net = graspn(bbone, numa)
        self.net.load_state_dict(torch.load(wpath, map_location="cpu"))
        self.net.eval()

    def display(self, idx):
        dataX, dataY = self.__getitem__(idx)
        dataX = dataX.reshape(-1, 3, ds.iwidth, ds.iwidth)
        preg, pcls = self.net(dataX)
        preds = pcls[:, 1] - pcls[:, 0]
        _, inds = torch.sort(preds , descending=True)
        
        
