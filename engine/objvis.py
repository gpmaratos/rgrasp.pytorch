from image_visualization.vis_obj import CGObject, show_obj

def show_objects(d_path):
    ds = CGObject(d_path)
    print('Object Number')
    for ind, obj in enumerate(ds):
        print('\t%d'%(ind))
        show_obj(obj)
