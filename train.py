import argparse
import math
import torch
from CornellUtils import CornellDataset
from GraspNetwork import graspn
from torch import optim
from torch.nn import functional as F

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dpath", type=str, help="path to the data")
    parser.add_argument("bbone", type=str, help="backbone type")
    parser.add_argument("angles", type=int, help="number of angles or k")
    parser.add_argument("-a", help="perform image augmentation",
    action="store_true")
    args = parser.parse_args()
    datap = args.dpath
    backb = args.bbone
    numa = args.angles
    augs = args.a

    #NEED SOME CHECKS ON THE ARGS

    dev = torch.device("cuda:0")
    ds = CornellDataset(datap, augs, numa)
    net = graspn(backb, numa).to(dev)

    lr = 1e-4
    opt = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4,\
        momentum=0.9)

    print("Begin Training")
    for epoch in range(1, 100):
        lravg, lcavg = [], []
        for ex in range(100):
            #Forward Pass
            dataX, dataY = ds[ex]
            dataX = dataX.reshape(-1, 3, ds.iwidth, ds.iwidth).to(dev)
            preg, pcls = net(dataX)
            #Calculate Loss
            tgs = torch.Tensor(dataY)
            inds = tgs[:, -3:].long()
            finds = inds[:, -1]*ds.anum**2 + inds[:, -3]*ds.anum\
                + inds[:, -2]
            tgr = tgs[:, :-3].to(dev)
            lreg = F.smooth_l1_loss(preg[finds], tgr)
            prec = pcls[finds]
            conf = F.softmax(pcls[:, 1], dim=0)
            conf[finds] = 0
            _, ninds = torch.sort(conf, descending=True)
            nrec = pcls[ninds[:3*len(finds)]]
            tcls = torch.zeros(4*len(finds), dtype=torch.long).to(dev)
            tcls[:len(finds)] = 1
            prcls = torch.cat((prec, nrec))
            lcls = F.cross_entropy(prcls, tcls)
            loss = (lcls + 2*lreg) / 4*len(finds)
            with torch.no_grad():
                lravg.append(lreg)
                lcavg.append(lcls)
            opt.zero_grad()
            loss.backward()
            opt.step()

        travg = torch.Tensor(lravg)
        tcavg = torch.Tensor(lcavg)
        print("%d: reg ~ %.2f, %.2f  cls ~ %.2f, %.2f"%(epoch,\
            torch.mean(travg), torch.std(travg),\
            torch.mean(tcavg), torch.std(tcavg)))

    import pdb;pdb.set_trace()
