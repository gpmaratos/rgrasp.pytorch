from torch import nn
from torch.nn import Sequential
from torchvision.models import resnet50

def build_backbone(cfg):
    backbone = ResnetBackbone()
    return backbone

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
