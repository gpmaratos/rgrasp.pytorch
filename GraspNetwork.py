from torch import nn
from torch.nn import Sequential
from torch.nn import Conv2d
from torchvision.models import resnet50

class graspn(nn.Module):
    def __init__(self, smodel, nang):
        super(graspn, self).__init__()
        if smodel == "resnet50":
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
            self.netr = Conv2d(2048, 5*nang, 3, padding=1)
            self.netcp = Conv2d(2048, nang, 3, padding=1)
            self.netcn = Conv2d(2048, nang, 3, padding=1)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        r = self.netr(x)
        p = self.netcp(x)
        n = self.netcn(x)
        return r, p, n
