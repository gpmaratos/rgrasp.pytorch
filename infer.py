import argparse
from utils.config import build_config
from engine.inference import infer
from engine.gt import gt
from models.general_rcnn import build_RCNN

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder (Cornell Grasp)')
    parser.add_argument('CPATH', type=str,
        help='path to configuration file (see utils/config for details)')
    parser.add_argument('WPATH', type=str,
        help='path to weights of network to be used in inference')
    parser.add_argument('-g', help='view gt', action='store_true')
    args = parser.parse_args()

    cfg = build_config(args.DPATH, args.CPATH, args.WPATH)

    if args.g:
        gt(cfg)
        return

    model = build_RCNN(cfg)
    model.load_state_dict(
        torch.load(args.inf, map_location='cpu')
    )
    infer(model, cfg)

main()
