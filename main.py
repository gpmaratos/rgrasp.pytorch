import torch
import argparse
from models.general_rcnn import build_RCNN
from visualize.vis import build_visualizer
from utils.config import build_config
from engine.trainer import train
from engine.visualizer import visualize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
        help='train with given path to data')
    parser.add_argument('--vis', type=str,
        help='visualize with given path to data')
    args = parser.parse_args()

    b_size = 15

    print('Beginning Training')
    if torch.cuda.is_available():
        dev = torch.device('cuda:1')
        print('Cuda detected, using GPU')
    else:
        dev = torch.dev('cpu')
        print('No cuda detected, using CPU')

    nn_cfg = {
        'bbone_type':'resnet50',
        'num_ang':4,
        'reg_features':100,
        'cls_features':100,
        'balance_factor':2,
        'alpha':2.,
    }

    if args.train:
        dpath = args.train
        cfg = build_config(dpath, dev, b_size, nn_cfg)
        model = build_RCNN(cfg)
        train(model, cfg, end=1000)

    if args.vis:
        dpath = args.vis
        cfg = build_config(dpath, dev, b_size, nn_cfg)
        visualizer = build_visualizer(dpath)
        visualize(visualizer, cfg)

if __name__ == "__main__":
    main()
