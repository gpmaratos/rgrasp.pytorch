from torch import nn
from torch.nn import Conv2d
from torch import sigmoid
from models.head.loss import Loss
from utils.balanced_sampler import build_balanced_sampler


def build_head(cfg):
    return CombinedHead(cfg)

class CombinedHead(nn.Module):
    """
    CombinedHead. Class which defines a network that predicts whether
    or not a sucessful grasp lies in a certain anchor point. Takes, as
    input, the feature maps from the backbone and returns detections/loss.

    Arguments:
        cfg (dictionairy): configuration file that specifies details of
            the architecture. see GraspHead for more details
    """

    def __init__(self, cfg):
        super(CombinedHead, self).__init__()
        reg_feat = cfg.reg_feat
        layer0 = Conv2d(2048, reg_feat, 3, padding=1)
        layer1 = Conv2d(reg_feat, 4*cfg.num_ang, 1)
        b_sampler = build_balanced_sampler(cfg)
        sigm = sigmoid
        loss_fun = Loss(cfg)
        self.layer0 = layer0
        self.layer1 = layer1
        self.b_sampler = b_sampler
        self.sigm = sigm
        self.loss_fun = loss_fun

    def forward(self, features, targets=None):
        x = self.layer0(features)
        x = self.layer1(x)
        x = self.sigm(x)
        if self.training:
            p_reg, p_cls, t_reg, t_cls, P = self.b_sampler(x, targets)
            loss = self.loss_fun(p_reg, p_cls, t_reg, t_cls, P)
            return x, loss
        return x, None

