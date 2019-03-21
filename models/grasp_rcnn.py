import torch
from random import randint
from torch import sigmoid
from torch import nn
from torch.nn import Sequential, Conv2d
from torchvision.models import resnet50
from data_processing.gt_extractor import GTExtractor
from torch.nn.functional import smooth_l1_loss, binary_cross_entropy

def build_model(device):
    return GeneralRCNN(device)

class GeneralRCNN(nn.Module):
    """
    GeneralRCNN. class that contains all the modules of the grasp network.
    Depending on the mode (training or testing) the output will have the
    calculated loss.

    Arguments:
        device (torch.device) device where memory for this object will live on

    input:
        (batch size) X (rgb_channels) X (img_dim_1) X (img_dim_2)
    output:
        predictions, loss
    """

    def __init__(self, device):
        super(GeneralRCNN, self).__init__()
        backbone = ResnetBackbone()
        head = CombinedHead(device, n_ang=4, h_feat=100)
        self.head = head
        self.backbone = backbone
        self.device = device
        self.to(device)

    def forward(self, img, targets=None):
        img = img.to(self.device)
        features = self.backbone(img)
        preds, loss = self.head(features, targets)
        return preds, loss

class CombinedHead(nn.Module):
    """
    CombinedHead. class that takes the feature map from the backbone as input
    and predicts grasping locations and offsets from the anchors.

    Arguments:
        device (torch.device) device where memory for this object will live on
        n_ang (int): number of angles represented in the anchors
        h_feat (int): size of the hiddent layer that processes the imagenet
            feature map
    """

    def __init__(self, device, n_ang=4, h_feat=100):
        super(CombinedHead, self).__init__()
        layer0 = Conv2d(2048, h_feat, 3, padding=1)
        layer1 = Conv2d(h_feat, 4*n_ang, 1)
        b_sampler = BalancedSampler(device, n_ang, b_factor=2)
        loss_fun = Loss(b_factor=2, alpha=1)
        self.layer0 = layer0
        self.layer1 = layer1
        self.b_sampler = b_sampler
        self.sigm = sigmoid
        self.loss_fun = loss_fun

    def forward(self, features, targets=None):
        x = self.layer0(features)
        x = self.layer1(x)
        x = self.sigm(x)
        x = x.permute(0, 2, 3, 1)
        if self.training:
            p_reg, p_cls, t_reg, t_cls, P = self.b_sampler(x, targets)
            loss = self.loss_fun(p_reg, p_cls, t_reg, t_cls, P)
            return x, loss
        return x, None

class ResnetBackbone(nn.Module):
    """
    ResnetBackbone. Class that defines a headless resnet50 (currently only
    supported network), which produces a feature map.
    """
    def __init__(self):
        super(ResnetBackbone, self).__init__()
        backb = resnet50(pretrained=True)
        self.layer0 = Sequential(\
            backb.conv1,\
            backb.bn1,\
            backb.relu,\
            backb.maxpool)
        self.layer1 = backb.layer1
        self.layer2 = backb.layer2
        self.layer3 = backb.layer3
        self.layer4 = backb.layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class BalancedSampler:
    """
    BalancedSampler. Class which takes network predictions and targets,
    and returns new predictions and targets which are tensors that are
    balanced (positive and negative) subsamples. The magic number 4
    comes from the fact that a prediction involves positions x, y, t
    and a classification score.

    Arguments:
        device (torch.device): device where input data should live on
        b_factor (int): balance factor that determines the ratio of
            positive to negative samples from a single example.
    """
    def __init__(self, device, n_ang, b_factor=2):
        super(BalancedSampler, self).__init__()
        gt_extractor = GTExtractor(n_ang)
        stride_factor = torch.tensor([gt_extractor.pixel_stride,
            gt_extractor.pixel_stride, gt_extractor.angle_stride]).to(device)
        self.gt_extractor = gt_extractor
        self.stride_factor = stride_factor
        self.pos_inds = []
        self.pos_examples = 0
        self.dev = device
        self.b_factor = b_factor

    def clear_state(self):
        self.pos_inds = []
        self.pos_examples = 0

    def _extract_predictions(self, preds, targs):
        #builds pos_inds which is needed for extract negative
        inds = [(i, j, k) for i, j, k in targs[0]]
        self.pos_inds.append(inds)
        preds = [preds[i, j, 4*k:4*k+4] for i, j, k in inds]
        preds = torch.stack(preds)
        preds_reg = preds[:, :3]
        preds_cls = preds[:, 3]
        self.pos_examples += len(preds_cls)
        return preds_reg*self.stride_factor, preds_cls

    def extract_predictions(self, predictions, targets):
        #creates balanced samples of regression and cls
        preds = [self._extract_predictions(predictions[i], targets[i])\
                for i in range(len(targets))]
        preds_cls_neg = [
            self._extract_negative(predictions[i], self.pos_inds[i])
            for i in range(len(self.pos_inds))
        ]
        preds_reg = torch.cat([pred[0] for pred in preds])
        preds_cls = [pred[1] for pred in preds]
        preds_cls += preds_cls_neg
        preds_cls = torch.cat(preds_cls)
        return preds_reg, preds_cls

    def _extract_negative(self, preds, targs):
        #I will sample random values not in the positive examples
        #i suspect this is the slowest part of the algorithm
        inds = []
        while len(inds) < self.b_factor*len(targs):
            sample = (randint(0, preds.shape[0]-1),
                        randint(0, preds.shape[1]-1),
                        randint(0, (preds.shape[2]//4)-1))
            if sample not in targs:
                inds.append(sample)
        preds_cls_neg = [preds[i, j, 4*k+3] for i, j, k in inds]
        preds_cls_neg = torch.stack(preds_cls_neg)
        return preds_cls_neg

    def build_targets(self, targets):
        targ_reg = [torch.tensor(target[1]) for target in targets]
        targ_reg = torch.cat(targ_reg, dim=0)
        targ_cls = torch.zeros(self.pos_examples+\
            self.pos_examples*self.b_factor)
        targ_cls[:self.pos_examples] = 1.
        return targ_reg, targ_cls

    def __call__(self, predictions, targets):
        targets = [self.gt_extractor(target) for target in targets]
        self.clear_state()
        preds_reg, preds_cls = self.extract_predictions(
                                    predictions, targets)
        targ_reg, targ_cls = self.build_targets(targets)
        return preds_reg, preds_cls,\
            targ_reg.to(self.dev), targ_cls.to(self.dev),\
            self.pos_examples

class Loss:
    """
    Loss metric. Calculates the final loss.
    """
    def __init__(self, b_factor, alpha):
        self.b_factor = b_factor
        self.alpha = alpha

    def __call__(self, p_reg, p_cls, t_reg, t_cls, P):
        reg_loss = smooth_l1_loss(p_reg, t_reg)
        cls_loss = binary_cross_entropy(p_cls, t_cls)
        final_loss = (cls_loss + self.alpha*reg_loss)/\
                        (P + P*self.b_factor)
        return final_loss

