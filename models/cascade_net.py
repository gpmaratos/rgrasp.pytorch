"""
The architecture corresponding to the cascaded approach in Deep Learning for
Detecting Robotic Grasps. A network is trained to detect the location of grasps,
and then a set of features are used to predict offsets
"""

import torch
from torch import nn
from torchvision.models import resnet50

def build_model(device):
    return CascadeNet(device).to(device)

class CascadeNet(nn.Module):

    def __init__(self, device):
        super(CascadeNet, self).__init__()
#        detector = CascadeDetector()
        detector = CascadeDetectorPretrained()
        predictor = CascadePredictor()
        self.device = device
        self.detector = detector
        self.predictor = predictor

    def forward(self, img_arr, img_lbl):
        img_arr = img_arr.to(self.device)
        detections, loss = self.detector(img_arr, img_lbl)
        return detections, loss
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

    def __init__(self):
        super(CascadeDetector, self).__init__()

        #high granularity features from image, potentially edge detectors
        base = 25
        self.high_gran_l0 = nn.Conv2d(3, base, (3, 3))

        self.l1 = nn.Conv2d(base, base, (13, 13))
        self.l2 = nn.Conv2d(base*2, base*2, (10, 10))
        self.l3 = nn.Conv2d(base*4, base*4, (5, 5))
        self.l4 = nn.Conv2d(base*8, base*8, (3, 3))

        #layer 5 is the predictive head
        self.l5 = nn.Conv2d(base*8, 1, (1, 1))

        self.bn = nn.BatchNorm2d(20)
        self.pl1 = AdaptiveConcatPool2d(153)
        self.pl2 = AdaptiveConcatPool2d(72)
        self.pl3 = AdaptiveConcatPool2d(34)

        self.relu = torch.nn.ReLU()
        self.lss = torch.nn.BCEWithLogitsLoss(reduction='mean')

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
        #each anchor is 10 pixels here
        loss = self.compute_loss(x, img_lbl)
        return x, loss

    def compute_loss(self, x, img_lbl):
        #assumed that x is of shape [batch, 1, 32, 32]
        inp, targ = [], []
        for i in range(len(img_lbl)):
            #extract positive and negative inds
            pos_inds = [(r.x_pos, r.y_pos) for r in img_lbl[i]]
            neg_size = len(pos_inds)*2
            cls_sort, cls_ind = torch.sort(x[i].view(-1), descending=True)
            j, neg_inds = 0, []
            while len(neg_inds) < neg_size:
                ind = cls_ind[j].item()
                pair = (int(ind // 32), int(ind % 32))
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
        return loss

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
        loss = self.compute_loss(x, img_lbl)
        return x, loss

    def compute_loss(self, x, img_lbl):
        #assumed that x is of shape [batch, 1, 32, 32]
        inp, targ = [], []
        for i in range(len(img_lbl)):
            #extract positive and negative inds
            pos_inds = [(r.x_pos, r.y_pos) for r in img_lbl[i]]
            neg_size = len(pos_inds)*2
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
        return loss
#inp = torch.ones(2, 4, 320, 320)
#model = CascadeNet(torch.device('cpu'))
#model(inp)
