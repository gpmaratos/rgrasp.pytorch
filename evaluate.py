import argparse
from EvalUtils import Display
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dpath", type=str, help="path to the data")
    parser.add_argument("params", type=str, help="path to weight file")
    parser.add_argument("bbone", type=str, help="backbone type")
    parser.add_argument("angles", type=int, help="number of angles or k")
    args = parser.parse_args()
    datap = args.dpath
    weightp = args.params
    backb = args.bbone
    numa = args.angles

    #NEED TO ADD ARG CHECKS

    disp = Display(datap, backb, weightp, numa)
    for i in range(100):
        disp.display(i)
