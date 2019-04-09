import torch
from torch import nn
from torchvision.models import resnet50
from target_formatting.five_tuple import BalancedSampler, Loss

def build_model(device):
    return GeneralRCNN(device).to(device)

class GeneralRCNN(nn.Module):

    def __init__(self, device):
        super(GeneralRCNN, self).__init__()
        backbone = ResnetBackbone()
        head = HeadNetwork()
        b_sampler = BalancedSampler()
        self.backbone = backbone
        self.head = head
        self.device = device
        self.b_sampler = b_sampler
        self.loss = Loss()

    def forward(self, img_batch, targets=None):
        img_batch = img_batch.to(self.device)
        features = self.backbone(img_batch)
        preds = self.head(features)
        if self.training:
            target_off, target_cls, pred_off, pred_cls = \
                self.b_sampler(preds, targets)
            target_off = target_off.to(self.device)
            target_cls = target_cls.to(self.device)
            loss ,off, cls = self.loss(target_off, target_cls, pred_off, pred_cls)
            return preds, loss, off, cls
        return preds, None

class ResnetBackbone(nn.Module):
    """
    ResnetBackbone. Class that defines a headless resnet50 (currently only
    supported network). Returns a set of feature maps at different scales.
    """
    def __init__(self):
        super(ResnetBackbone, self).__init__()
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

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x2, x3, x4, x5

class HeadNetwork(nn.Module):
    """
    Takes a set of feature maps and predicts grasps
    throwing out the highest level layer (later I can add upsampling)
    so that I can have 16 width anchors.
    """

    def __init__(self):
        super(HeadNetwork, self).__init__()
        outf = 100
        self.layer0 = nn.Conv2d(256, outf, (61, 61))
        self.bn0 = nn.BatchNorm2d(outf)
        self.layer1 = nn.Conv2d(512, outf, (21, 21))
        self.bn1 = nn.BatchNorm2d(outf)
#        self.layer2 = nn.Conv2d(1024, outf, (11, 11))
#        self.bn2 = nn.BatchNorm2d(outf)
#        self.layer3 = nn.Conv2d(2048, outf, 3, padding=1)
#        self.bn3 = nn.BatchNorm2d(outf)

        self.layer2 = nn.Conv2d(1024, outf, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outf)

        self.relu = nn.ReLU(inplace=True)
        #next layer predicts offsets and 2 cls layers
        self.layer4 = nn.Conv2d(outf*3, 7, 3, padding=1)

    def forward(self, feature_maps):
        x1 = self.layer0(feature_maps[0])
        x1 = self.bn0(x1)
        x2 = self.layer1(feature_maps[1])
        x2 = self.bn1(x2)
        x3 = self.layer2(feature_maps[2])
        x3 = self.bn2(x3)
#        x4 = self.layer3(feature_maps[3])
#        x4 = self.bn3(x4)
#        features = torch.cat((x1, x2, x3, x4), 1)
        features = torch.cat((x1, x2, x3), 1)
        features = self.relu(features)
        prediction = self.layer4(features)
        self.relu(prediction[:, -2:, :, :])
        return prediction

#inp = torch.ones(4, 3, 320, 320)
#bbone = ResnetBackbone()
#head = HeadNetwork()
#sampler = BalancedSampler()
#out = bbone(inp)
#out1 = head(out, None)

#torch.Size([1, 256, 80, 80])
#torch.Size([1, 512, 40, 40])
#torch.Size([1, 1024, 20, 20])
#torch.Size([1, 2048, 10, 10])
