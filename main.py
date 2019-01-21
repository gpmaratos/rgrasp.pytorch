import argparse
from models.general_rcnn import build_RCNN
from utils.config import build_config
from engine.trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
        help='train with given path to data')
    args = parser.parse_args()

    dpath = args.train
    b_size = 10
    nn_cfg = {
        'bbone_type':'resnet50',
        'num_ang':4,
        'reg_features':100,
        'cls_features':100,
    }
    cfg = build_config(dpath, b_size, nn_cfg)

    if args.train:
        model = build_RCNN(cfg)
        train(model, cfg)

if __name__ == "__main__":
    main()
