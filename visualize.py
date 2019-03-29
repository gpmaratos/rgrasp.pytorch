import argparse
from engine.visualizer import visualize

def main():
    """
    Visualize the input data one object at a time, and see the augmentations

    Command Line Arguments:
        DPATH: path to the folder containing the data, z.txt, and background imgs
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    visualize(d_path)

main()
