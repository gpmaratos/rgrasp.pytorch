"""
The architecture corresponding to the cascaded approach in Deep Learning for
Detecting Robotic Grasps. A network is trained to detect the location of grasps,
and then a set of features are used to predict offsets
"""

import torch
from torch import nn
from itertools import product
from torchvision.models import vgg16_bn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from random import randint

def build_model(device):
    return CascadeNet(device).to(device)

class CascadeNet(nn.Module):

    def __init__(self, device):
        super(CascadeNet, self).__init__()
#        detector = CascadeDetector(device)
        detector = CascadeDetectorPretrained(device)
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
        base = 20
        self.lss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))
        self.layer0 = nn.Conv2d(3, base, (3, 3))
        self.layer1 = nn.Conv2d(base, base, (3, 3))
        self.layer2 = nn.Conv2d(base, base, (3, 3))
        self.layer3 = nn.Conv2d(base, base, (3, 3))
        self.layer4 = nn.Conv2d(base, base, (3, 3))
        self.predictive_head = nn.Conv2d(base, 2, (1, 1))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU(inplace=True)
        self.device = device

    def forward(self, x, y):
        x = self.layer0(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.predictive_head(x)
        if self.training:
            loss, prec, rec, f1, sup = self.compute_loss(x, y)
            return x, loss, prec, rec, f1, sup
        else:
            return x, None, None, None, None, None

    def compute_loss(self, x, img_lbl):
        #assumed that x is of shape [batch, 1, 32, 32]
        inp, targ = [], []
        for i in range(len(img_lbl)):
            #extract positive and negative inds
            pos_inds = [(r.x_pos, r.y_pos) for r in img_lbl[i]]
            neg_size = len(pos_inds)
            #find best false positives
            differences = []
            for p_x, p_y in product(range(10), range(10)):
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
        prec, rec, f1, sup = precision_recall_fscore_support(targs, inps, average='binary')
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

    def __init__(self, device):
        super(CascadeDetectorPretrained, self).__init__()
        backb = vgg16_bn(pretrained=True)
        self.layer0 = backb.features
        self.layer1 = nn.Conv2d(512, 2, (1, 1))
        self.device = device
        self.lss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]))

    def forward(self, x, img_lbl):
        x = self.layer0(x)
        x = self.layer1(x)
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
            #find best false positives
            differences = []
            for p_x, p_y in product(range(10), range(10)):
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
        prec, rec, f1, sup = precision_recall_fscore_support(targs, inps, average='binary')
        return loss, prec, rec, f1, sup

