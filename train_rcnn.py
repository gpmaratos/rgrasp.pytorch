import argparse
from engine.trainer import train

def main():
    """
    Do inference on the Cornell Grasp dataset. Trains model using GD.

    Command Line Arguments:
        DPATH: path to the folder containing the data, z.txt, and background imgs
        WPATH: path to folder which the model is saved after training
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    parser.add_argument('WPATH', type=str,
        help='path to weights folder (models are saved here)')
    args = parser.parse_args()

    d_path = args.DPATH
    w_path = args.WPATH
    batch_size = 3
    train(d_path, w_path, batch_size)

main()
