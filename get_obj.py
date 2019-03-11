import argparse
from engine.objvis import show_objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder (Cornell Grasp)')
    args = parser.parse_args()
    show_objects(args.DPATH)

main()
