"""
Script to report:
    # of objects
    # of images
    # of rectangles
"""

import argparse
from data_processing.dataset import CornellDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds_train = CornellDataset(d_path, 'train')
    ds_val = CornellDataset(d_path, 'val')
    ds_test = CornellDataset(d_path, 'test')

    object_nums = [len(ds_train), len(ds_val), len(ds_test)]
    img_nums, rec_nums = [], []
    for ds in [ds_train, ds_val, ds_test]:
        img_num, rec_num = 0, 0
        for bind, batch in enumerate(ds):
            print(' %d/%d '%(bind+1, len(ds)), end='\r')
            img_num += len(batch)
            for bh in batch:
                rec_num += len(bh[2])
        img_nums.append(img_num)
        rec_nums.append(rec_num)
        print('')

    print('Counts')
    print('\tobjects\timages\trecs')
    for typ in [('train', 0), ('val', 1), ('test', 2)]:
        print('%s\t%d\t%d\t%d'%(typ[0], object_nums[typ[1]],
            img_nums[typ[1]], rec_nums[typ[1]]))
    print('totals\t%d\t%d\t%d'%(sum(object_nums),
        sum(img_nums), sum(rec_nums)))
main()
