from torch import nn
from models.backbone import build_backbone
from models.combined_head import build_head

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
        head = build_head(cfg)
        self.head = head
        self.backbone = backbone

    def forward(self, img):
        #loss calculation needs fixing (need to calculate final loss)
        features = self.backbone(img)
        #needs fixing for ghead
        return reg_scores, reg_loss, cls_scores, cls_loss
