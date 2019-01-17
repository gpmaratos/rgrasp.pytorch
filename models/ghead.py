from torch import nn

class GraspHead(nn.Module):
    """
    GraspHead. Class which defines a network that predicts whether
    or not a sucessful grasp lies in a certain anchor point. Takes, as
    input, the feature maps from the backbone and returns detections/loss.

    Arguments:
        cfg (dictionairy): configuration file that specifies details of
            the architecture. see GraspHead for more details
    """

    def __init__(self, cfg):
