from data_processing.dataset import build_dataset
from image_visualization.vis_obj import show_obj

def show_objects(cfg):
    ds = build_dataset(cfg, train=False, aug=False, dset='train')
    print('Object Number')
    for ind, obj in enumerate(ds):
        print('\t%d'%(ind))
        obj = [x[0] for x in obj]
        show_obj(obj)
