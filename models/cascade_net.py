"""
The architecture corresponding to the cascaded approach in Deep Learning for
Detecting Robotic Grasps. A network is trained to detect the location of grasps,
and then a set of features are used to predict offsets
"""

import torch
from torch import nn
from itertools import product
from torchvision.models import resnet50
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from random import randint

def build_model(device):
    return CascadeNet(device).to(device)

class CascadeNet(nn.Module):

    def __init__(self, device):
        super(CascadeNet, self).__init__()
        detector = CascadeDetector(device)
#        detector = CascadeDetectorPretrained()
        predictor = CascadePredictor()
        self.device = device
        self.detector = detector
        self.predictor = predictor

    def forward(self, img_arr, img_lbl):
        img_arr = img_arr.to(self.device)
        detections, loss, prec, rec, f1, sup = self.detector(img_arr, img_lbl)

        result_dict = {'detections':detections, 'loss':loss,
            'prec':prec, 'rec':rec, 'f1':f1, 'sup':sup,
        }
        return detections, result_dict
        """ The model needs to run inference and report the loss for
            both detector and predictor """

class CascadeDetector(nn.Module):
    """
    The detector should take a form similar to grasp rcnn. It finds graspable
    locations from an image. This is done in a fully convolutional way, and
    the loss is calculated via cross entropy.

    The input will be a 4 channel view of the image. Features are extracted
    through a feature pyramid.

    If the goal is to achieve 20 x 20 anchor boxes, then the original image
    needs to be reduced by 300 pixels
    """

    def __init__(self, device):
        super(CascadeDetector, self).__init__()

        #high granularity features from image, potentially edge detectors
        base = 55
        self.high_gran_l0 = nn.Conv2d(3, base, (3, 3))

        self.l1 = nn.Conv2d(base, base, (13, 13))
        self.l2 = nn.Conv2d(base*2, base*2, (10, 10))
        self.l3 = nn.Conv2d(base*4, base*4, (5, 5))
        self.l4 = nn.Conv2d(base*8, base*8, (3, 3))

        #layer 5 is the predictive head
        self.l5 = nn.Conv2d(base*8, 2, (1, 1))

        self.bn = nn.BatchNorm2d(20)
        self.pl1 = AdaptiveConcatPool2d(153)
        self.pl2 = AdaptiveConcatPool2d(72)
        self.pl3 = AdaptiveConcatPool2d(34)

        self.relu = torch.nn.ReLU()
#        self.lss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.device = device
        self.lss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.2]))

    def forward(self, img_arr, img_lbl):
        x = self.high_gran_l0(img_arr)
        x = self.relu(x)

        x = self.l1(x)
        x = self.relu(x)
        x = self.pl1(x)

        x = self.l2(x)
        x = self.relu(x)
        x = self.pl2(x)

        x = self.l3(x)
        x = self.relu(x)
        x = self.pl3(x)

        x = self.l4(x)
        x = self.relu(x)

        x = self.l5(x)
        x = self.relu(x)
        #each anchor is 10 pixels here
        if self.training:
            loss, prec, rec, f1, sup = self.compute_loss(x, img_lbl)
            return x, loss, prec, rec, f1, sup
        else:
            return x, None, None, None, None, None

    def compute_loss(self, x, img_lbl):
        #assumed that x is of shape [batch, 1, 32, 32]
        inp, targ = [], []
        for i in range(len(img_lbl)):
            #extract positive and negative inds
            pos_inds = [(r.x_pos, r.y_pos) for r in img_lbl[i]]
            neg_size = len(pos_inds)*3

            #finding the points the network is most confidently incorrect
            differences = []
            for p_x, p_y in product(range(32), range(32)):
                if x[i, 0, p_x, p_y] < x[i, 1, p_x, p_y]:
                    diff = x[i, 1, p_x, p_y] - x[i, 0, p_x, p_y]
                    differences.append((diff, p_x, p_y))
            sorted_differences = sorted(differences, key=lambda z:z[0], reverse=True)
            neg_inds = []
            j = 0
            if len(sorted_differences) > 0:
                while j < neg_size:
                    if not sorted_differences[j][1:] in pos_inds:
                        neg_inds.append(sorted_differences[j][1:])
                    j += 1
                    if j > len(sorted_differences)-1:
                        break
            if len(neg_inds) < neg_size:
                while len(neg_inds) < neg_size:
                    pair = (randint(0, x.shape[2]-1), randint(0, x.shape[3]-1))
                    if not pair in pos_inds:
                        neg_inds.append(pair)

            #build array
            for pair in pos_inds:
                inp.append(x[i, :, pair[0], pair[1]])
                targ.append(1)
            for pair in neg_inds:
                inp.append(x[i, :, pair[0], pair[1]])
                targ.append(0)
        #compute cross entropy loss
        inp_tensor = torch.stack(inp)
        targ_tensor = torch.tensor(targ).to(self.device)
        loss = self.lss(inp_tensor, targ_tensor)
        #I will use a threshold of 0.5 but it might not be optimal
        inps = np.array([0 if t[0] > t[1] else 1 for t in inp])
        targs = targ_tensor.cpu().numpy()
        prec, rec, f1, sup = precision_recall_fscore_support(targs, inps)
        return loss, prec, rec, f1, sup

    def compute_loss_alt(self, x, img_lbl):
        pos_inds = []
        targ_tensor = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], dtype=torch.long)
        for i in range(len(img_lbl)):
            for j in range(len(img_lbl[i])):
                targ_tensor[i, 0, img_lbl[i][j].x_pos, img_lbl[i][j].y_pos] = 1
        x = x.permute(0, 2, 3, 1)
        targ_tensor = targ_tensor.permute(0, 2, 3 ,1).to(self.device)
        x = x.reshape(-1, 2)
        targ_tensor = targ_tensor.view(-1)
        #I should double check these tensors align
        loss = self.lss(x, targ_tensor)
        inps = np.array([0 if x[i, 0] > x[i, 1] else 1 for i in range(len(x))])
        targs = targ_tensor.cpu().numpy()
        prec, rec, f1, sup = precision_recall_fscore_support(targs, inps)
        return loss, prec, rec, f1, sup

class CascadePredictor(nn.Module):
    def __init__(self):
        super(CascadePredictor, self).__init__()

class AdaptiveConcatPool2d(nn.Module):
    """
    AdaptiveConcatPool2d. Pooling layer that computes both AvgPooling
    and MaxPooling, then returns the concatenation. This improves
    performance beyond using one or the other
    Arguments:
        sz (int): the size of the pooling field for both avg and max
    """

    def __init__(self, sz=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)

class CascadeDetectorPretrained(nn.Module):

    def __init__(self):
        super(CascadeDetectorPretrained, self).__init__()
        backb = resnet50(pretrained=True)
        self.layer0 = nn.Sequential(\
            backb.conv1,\
            backb.bn1,\
            backb.relu,\
            backb.maxpool)
        self.layer1 = backb.layer1
        self.layer2 = backb.layer2
        self.layer3 = backb.layer3
        self.layer4 = backb.layer4
        self.lss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.layer5 = nn.Conv2d(1024, 1, (1, 1))

    def forward(self, x, img_lbl):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer5(x)
        loss, prec, rec, f1, sup = self.compute_loss(x, img_lbl)
        return x, loss, prec, rec, f1, sup

    def compute_loss(self, x, img_lbl):
        inp, targ = [], []
        for i in range(len(img_lbl)):
            #extract positive and negative inds
            pos_inds = [(r.x_pos, r.y_pos) for r in img_lbl[i]]
            neg_size = len(pos_inds)*1.5
            cls_sort, cls_ind = torch.sort(x[i].view(-1), descending=True)
            j, neg_inds = 0, []
            while len(neg_inds) < neg_size:
                ind = cls_ind[j].item()
                pair = (int(ind // 20), int(ind % 20))
                if pair in pos_inds:
                    j += 1
                    continue
                neg_inds.append(pair)
                j += 1
            #build array
            for pair in pos_inds:
                inp.append(x[i, 0, pair[0], pair[1]])
                targ.append(1.)
            for pair in neg_inds:
                inp.append(x[i, 0, pair[0], pair[1]])
                targ.append(0.)
        #compute binary cross entropy loss
        inp_tensor = torch.stack(inp).cpu()
        targ_tensor = torch.tensor(targ)
        loss = self.lss(inp_tensor, targ_tensor)
        #I will use a threshold of 0.5 but it might not be optimal
        inps = (torch.sigmoid(inp_tensor) > 0.5).numpy()
        targs = targ_tensor.numpy()
        prec, rec, f1, sup = precision_recall_fscore_support(targs, inps)
        return loss, prec, rec, f1, sup

#    def compute_loss(self, x, img_lbl):
#        #assumed that x is of shape [batch, 1, 32, 32]
#        inp, targ = [], []
#        for i in range(len(img_lbl)):
#            #extract positive and negative inds
#            pos_inds = [(r.x_pos, r.y_pos) for r in img_lbl[i]]
#            neg_size = len(pos_inds)*2
#            cls_sort, cls_ind = torch.sort(x[i].view(-1), descending=True)
#            j, neg_inds = 0, []
#            while len(neg_inds) < neg_size:
#                ind = cls_ind[j].item()
#                pair = (int(ind // 20), int(ind % 20))
#                if pair in pos_inds:
#                    j += 1
#                    continue
#                neg_inds.append(pair)
#                j += 1
#            #build array
#            for pair in pos_inds:
#                inp.append(x[i, 0, pair[0], pair[1]])
#                targ.append(1.)
#            for pair in neg_inds:
#                inp.append(x[i, 0, pair[0], pair[1]])
#                targ.append(0.)
#        #compute binary cross entropy loss
#        inp_tensor = torch.stack(inp).cpu()
#        targ_tensor = torch.tensor(targ)
#        loss = self.lss(inp_tensor, targ_tensor)
#        return loss
#inp = torch.ones(2, 4, 320, 320)
#model = CascadeNet(torch.device('cpu'))
#model(inp)
