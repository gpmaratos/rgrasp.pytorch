import torch
import argparse
from utils.config import build_config
from models.general_rcnn import build_RCNN
from engine.trainer import train

def main():
    """
    Train a network. Configures the environment and calls the training
    function.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder (Cornell Grasp)')
    parser.add_argument('CPATH', type=str,
        help='path to configuration file (see utils/config for details)')
    parser.add_argument('WPATH', type=str,
        help='path to where the network will be saved when done training')
    args = parser.parse_args()

    cfg = build_config(args.DPATH, args.CPATH, args.WPATH)
    model = build_RCNN(cfg)
    train(model, cfg)

main()
