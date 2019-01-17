from torch import nn
from torch.nn import Conv2d, Linear
from loss.regression_loss import RegressionLoss

def build_regression_head(cfg):
    return RegressionHead(cfg)

class RegressionHead(nn.Module):
    """
    RegressionHead. Class which defines the head of the grasp network that
    predicts the location of grasps (as an offset from an anchor point),
    given a feature map as input. The magic number 2048 comes from the
    final dimension of the tensor that comes out of a Resnet50.

    Arguments:
        cfg (dictionairy): configuration file that specifies details of
            the architecture. see GeneralRCNN for more details
    """

    def __init__(self, cfg):
        super(RegressionHead, self).__init__()
        reg_feat = cfg['reg_features']
        layer0 = Conv2d(2048, reg_feat, 3, padding=1)
        layer1 = Linear(reg_feat, 3*cfg['num_ang'])
        self.reg_feat = reg_feat
        self.layer0 = layer0
        self.layer1 = layer1

    def forward(self, fmap, targets=None):
        x = self.layer0(fmap)
        x = x.view(x.shape[0], -1, self.reg_feat)
        x = self.layer1(x)

        if self.training:
            if not targets:
                raise ValueError('Targets needed in training mode')
            return x, RegressionLoss(x, targets)

        return x, None
