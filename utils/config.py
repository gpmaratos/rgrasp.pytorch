def build_config(dpath, dev, b_size, nn_cfg):
    return Configuration(dpath, dev, b_size, nn_cfg)

class Configuration:
    """
    Configuration. Class that encapsulates all the options for training
    the grasping network, all of which are passed during initialization.

    Arguments:
        dpath (string): path to the folder containing all of the
            example images

        dev (torch.device): object which defines the device where memory
            components will go

        b_size (int): how many examples will be part of a batch, which
            will depend on how much memory you wish to allocate at once
            to the input images

        nn_cfg (dict): configuration for the RCNN network which should have
            the following fields defined:

            bbone_type (string): defines which backbone to use

            num_ang (int): the number of angles at one position

            reg_feat (int): number of features for middle layer of
                regression head

            cls_feat (int): number of features for middle layer of
                classification head

            balance_factor (int): defines the factor of negative examples
                to positive examples. integers greater than or equal to
                one.

            alpha (float): weight factor on regression component of loss
                function
    """
    def __init__(self, dpath, dev, b_size, nn_cfg):
        self.dpath = dpath
        self.dev = dev
        self.b_size = b_size
        self.nn_cfg = nn_cfg
