from torch import nn
from torch.nn import Conv2d, Linear
from loss.class_loss import ClassLoss

def build_class_head(cfg):
    return ClassHead(cfg)

class ClassHead(nn.Module):
    """
    ClassHead. Class which defines the head of the grasp network that
    predicts whether or not a grasp is at a defined anchor point, given
    a feature map as input. The magic number 2048 comes from the final
    dimension of the tensor that comes out of a Resnet50.

    Arguments:
        cfg (dictionairy): configuration file that specifies details of
            the architecture. see utils/config.py for more information
    """

    def __init__(self, cfg):
        super(ClassHead, self).__init__()
        cls_feat = cfg['cls_features']
        layer0 = Conv2d(2048, cls_feat, 3, padding=1)
        layer1 = Linear(cls_feat, 2*cfg['num_ang'])
        self.cls_feat = cls_feat
        self.layer0 = layer0
        self.layer1 = layer1

    def forward(self, fmap, targets=None):
        x = self.layer0(fmap)
        x = x.view(x.shape[0], -1, self.cls_feat)
        x = self.layer1(x)

        if self.training:
            if not targets:
                raise ValueError('Targets needed in training mode')
            return x, ClassLoss(x, targets)

        return x, None

