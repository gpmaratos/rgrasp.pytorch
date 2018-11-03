import argparse
from EvalUtils import Display
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dpath", type=str, help="path to the data")
    parser.add_argument("params", type=str, help="path to weight file")
    parser.add_argument("bbone", type=str, help="backbone type")
    parser.add_argument("angles", type=int, help="number of angles or k")
    parser.add_argument("-a", help="use image augmentation",
    action="store_true")
    args = parser.parse_args()
    datap = args.dpath
    backb = args.bbone
    weightp = args.params
    numa = args.angles
    augs = args.a

    #NEED SOME CHECKS ON THE ARGS

    disp = Display(datap, backb, weightp, numa, augs)
    for i in range(100):
        disp.display(i)
