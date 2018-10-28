import argparse
import math
import torch
from CornellUtils import CornellDataset
from GraspNetwork import graspn
from torch import optim
from torch.nn import functional as F

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("-a", help="perform image augmentation"
    #action="store_true")
    #this will be used when I implement image transformations

    parser.add_argument("dpath", type=str, help="path to the data")
    parser.add_argument("bbone", type=str, help="backbone type")
    parser.add_argument("angles", type=int, help="number of angles or k")
    args = parser.parse_args()
    datap = args.dpath
    backb = args.bbone
    numa = args.angles

    #NEED SOME CHECKS ON THE ARGS

    dev = torch.device("cuda:0")
    ds = CornellDataset(datap, numa)
    net = graspn(backb, numa).to(dev)

    lr = 1e-4
    opt = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4)

    print("Begin Training")
    for epoch in range(1, 100):
        lravg, lcavg = [], []
        for ex in range(100):
            #Forward Pass
            dataX, dataY = ds[ex]
            dataX = dataX.reshape(-1, 3, ds.iwidth, ds.iwidth).to(dev)
            outr, outp, outn = net(dataX)
            #Calculate Loss
            lreg, lcls = torch.zeros(1).to(dev), torch.zeros(1).to(dev)
            pinds = []
            outpv = -1*F.log_softmax(outp.view(-1), dim=0)
            for t in dataY:
                aind = ds.get_anchor_ind(t[4])
                xpos = int((t[0]-ds.cXl) // ds.adim)
                ypos = int((t[1]-ds.cYu) // ds.adim)
                t = ((t[0] - xpos*ds.adim+(ds.adim/2))/ds.adim,\
                    (t[1] - ypos*ds.adim+(ds.adim/2))/ds.adim,\
                    math.log(t[2]/ds.adim), math.log(t[3]/ds.adim),\
                    (t[4] - ds.angs[aind])/(180/numa))
                t = torch.Tensor(t).reshape(-1, 5).to(dev)
                rp = outr[:, 5*aind:5*aind+5, xpos, ypos]
                #pindx = utils.tr(aind, xpos, ypos, outp)
                #NEEDS confirming
                pindx = aind*ds.anum**2 + xpos*ds.anum + ypos
                pinds.append(pindx)
                lreg += F.smooth_l1_loss(rp, t)
                lcls += outpv[pindx]
            length = 3*len(pinds)
            outuv = -1*F.log_softmax(outn.view(-1), dim=0)
            outpv[pinds] = 0
            _ , indicies = torch.sort(outpv, descending=True)
            lcls += torch.sum(outuv[indicies[:length]])
            loss = (lcls + 2*lreg) / 4*len(pinds)
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
