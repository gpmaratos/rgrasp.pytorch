import argparse
from structures.config import build_config
from engine.objvis import show_objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder (Cornell Grasp)')
    parser.add_argument('CPATH', type=str,
        help='path to configuration file')
    args = parser.parse_args()
    cfg = build_config(args.DPATH, args.CPATH, None)
    show_objects(cfg)

main()
