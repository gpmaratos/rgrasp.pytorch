import torch

def build_config(d_path, c_path, w_path):
    return Configuration(d_path, c_path, w_path)

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

        h_feat (int): number of features for middle layer of the head

        balance_factor (int): defines the factor of negative examples
            to positive examples. integers greater than or equal to
            one.

        alpha (float): weight factor on regression component of loss
            function
    """

    def read_conf_file(self, c_path):
        #algorithm is O(n^2) so don't have a large config file
        with open(c_path) as f:
            c_file = f.read().split('\n')[:-1]
        for field in c_file:
            field = field.split(':')
            if field[0] == 'b_size':
                self.b_size = int(field[1])
            if field[0] == 'n_ang':
                self.n_ang = int(field[1])
            if field[0] == 'h_feat':
                self.h_feat = int(field[1])
            if field[0] == 'b_factor':
                self.b_factor = int(field[1])
            if field[0] == 'alpha':
                self.alpha = float(field[1])

        if not hasattr(self, 'b_size'):
            self.b_size = 1
        if not hasattr(self, 'n_ang'):
            self.n_ang = 4
        if not hasattr(self, 'h_feat'):
            self.h_feat = 100
        if not hasattr(self, 'b_factor'):
            self.b_factor = 2
        if not hasattr(self, 'alpha'):
            self.alpha = 2.

    def print_config(self):
        print("Configuration:")
        print("\tb_size: %d"%(self.b_size))
        print("\tn_ang: %d"%(self.n_ang))
        print("\th_feat: %d"%(self.h_feat))
        print("\tb_factor: %d"%(self.b_factor))
        print("\talpha: %f"%(self.alpha))
        print("\tDevice: ", self.dev)
        print("\td_path: %s"%(self.d_path))
        print("\tw_path: %s\n"%(self.w_path))

    def __init__(self, d_path, c_path, w_path):
        self.d_path = d_path
        self.w_path = w_path
        self.read_conf_file(c_path)
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')
        self.print_config()
