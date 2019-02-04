import torch
import argparse
from models.general_rcnn import build_RCNN
from visualize.vis import build_visualizer
from utils.config import build_config
from engine.trainer import train
from engine.visualizer import visualize

def main():
    #FIX ORDERING SO CONFIG IS NOT CREATED TWICE
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath', type=str,
        help='path to data')
    parser.add_argument('-t',
        help='train a new model',
        action='store true')
    parser.add_argument('-i',
        help='run inference',
        action='store true')

    args = parser.parse_args()

    print('Beginning Training')
    if torch.cuda.is_available():
        dev = torch.device('cuda:1')
        print('Cuda detected, using GPU')
    else:
        dev = torch.device('cpu')
        print('No cuda detected, using CPU')

    cfg_dict = {
        'dpath':args.dpath,
        'dev':dev,
        'b_size':15,
        'bbone_type':'resnet50',
        'num_ang':4,
        'reg_features':100,
        'cls_features':100,
        'balance_factor':2,
        'alpha':2.,
    }

    cfg = build_config(cfg_dict)

    if args.train:
        model = build_RCNN(cfg)
        train(model, cfg)

if __name__ == "__main__":
    main()
