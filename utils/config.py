def build_config(cfg_dict):
    return Configuration(cfg_dict)

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

    def __init__(self, cfg_dict):
        self.dpath = cfg_dict['dpath']
        self.dev = cfg_dict['dev']
        self.b_size = cfg_dict['b_size']
        self.bbone_type = cfg_dict['bbone_type']
        self.num_ang = cfg_dict['num_ang']
        self.reg_feat = cfg_dict['reg_feat']
        self.cls_feat = cfg_dict['cls_feat']
        self.balance_factor = cfg_dict['balance_factor']
        self.alpha = cfg_dict['alpha']
