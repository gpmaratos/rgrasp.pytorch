from torch import nn
from models.backbone import build_backbone
from models.regression_head import build_regression_head
from models.class_head import build_class_head

def build_RCNN(cfg):
    return GeneralRCNN(cfg)

class GeneralRCNN(nn.Module):
    """
    GeneralRCNN. The model structure is based off the maskrcnn benchmark
    found in https://github.com/facebookresearch/maskrcnn-benchmark.
    It contains, in this case, two major components, the feature
    extracting backbone and the grasp predicting head.
    """
    def __init__(self, cfg):
        super(GeneralRCNN, self).__init__()
        backbone = build_backbone(cfg)
        regression_head = build_regression_head(cfg)
        class_head = build_class_head(cfg)
        self.backbone = backbone
        self.regression_head = regression_head
        self.class_head = class_head

    def forward(self, img):
        features = self.backbone(img)
        reg_scores, reg_loss = self.regression_head(features)
        cls_scores, cls_loss = self.class_head(features)
        return reg_scores, reg_loss, cls_scores, cls_loss
