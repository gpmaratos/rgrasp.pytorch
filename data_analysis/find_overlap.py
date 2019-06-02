import sys
import os
sys.path.insert(0, os.path.split(sys.path[0])[0])

import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

"""
Script to determine if overlap exists in the anchors
the result is that using an achor of 16 pixels is best
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds_train = CornellDataset(d_path, 'train')
    ds_val = CornellDataset(d_path, 'val')

    for ds in [ds_train, ds_val]:
        overlaps = []
        for bind, batch in enumerate(ds):
            print(' %d/%d   '%(bind+1, len(ds)), end='\r')
            for example in batch:
                temp = []
                count = 0
                for rec in example[2]:
                    pos = (rec.x_pos, rec.y_pos)
                    if pos in temp:
                        count += 1
                    else:
                        temp.append(pos)
                overlaps.append(count)
        plt.hist(overlaps)
        plt.show()

main()
