import torch
import argparse
from models.general_rcnn import build_RCNN
from engine.trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
        help='train with given path to data')
    args = parser.parse_args()
    dpath = args.train

    cfg = {
        'bbone_type':'resnet50',
        'num_ang':4,
        'reg_features':100,
        'cls_features':100,
    }

    if args.train:
        #run training loop
        dev = torch.dev('cuda:1')
        model = build_RCNN(cfg)


if __name__ == "__main__":
    main()
