"""
The architecture corresponding to the cascaded approach in Deep Learning for
Detecting Robotic Grasps. A network is trained to detect the location of grasps,
and then a set of features are used to predict offsets
"""

import torch
from torch import nn

def build_model(device):
    return CascadeNet(device).to(device)

class CascadeNet(nn.Module):

    def __init__(self, device):
        super(CascadeNet, self).__init__()
        detector = CascadeDetector()
        predictor = CascadePredictor()
        self.device = device
        self.detector = detector
        self.predictor = predictor

    def forward(self, img_arr, img_lbl):
        img_arr = img_arr.to(self.device)
        detections = self.detector(img_arr, img_lbl)
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
        self.high_gran_l0 = nn.Conv2d(6, 20, (3, 3))

        self.l1 = nn.Conv2d(20, 20, (13, 13))
        self.l2 = nn.Conv2d(40, 40, (10, 10))
        self.l3 = nn.Conv2d(80, 80, (5, 5))
        self.l4 = nn.Conv2d(160, 160, (3, 3))

        self.bn = nn.BatchNorm2d(20)
        self.pl1 = AdaptiveConcatPool2d(153)
        self.pl2 = AdaptiveConcatPool2d(72)
        self.pl3 = AdaptiveConcatPool2d(34)

    def forward(self, img_arr, img_lbl):
        x = self.high_gran_l0(img_arr)
        x = self.l1(x)
        x = self.pl1(x)
        x = self.l2(x)
        x = self.pl2(x)
        x = self.l3(x)
        x = self.pl3(x)
        x = self.l4(x)
        #each anchor is 10 pixels here
#        import pdb;pdb.set_trace()
        return x

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

#inp = torch.ones(2, 4, 320, 320)
#model = CascadeNet(torch.device('cpu'))
#model(inp)
