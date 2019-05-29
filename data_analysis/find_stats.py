"""
Script to report summary statistics on rectangle counts per image:
    mean
    median
    distribution
"""

import sys
import os
sys.path.insert(0, os.path.split(sys.path[0])[0])

import numpy as np
import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds_train = CornellDataset(d_path, 'train')
    ds_val = CornellDataset(d_path, 'val')
#    ds_test = CornellDataset(d_path, 'test')

    for ds in [ds_train, ds_val]:
        rec_counts = []
        for bind, batch in enumerate(ds):
            print(' %d/%d '%(bind+1, len(ds)), end='\r')
            [rec_counts.append(len(bh[2])) for bh in batch]
        rec_counts_np = np.array(rec_counts)
        mean = np.mean(rec_counts_np)
        median = np.median(rec_counts_np)
        plt.hist(rec_counts)

        print('')
        print('\tmean\tmedian')
        print('\t%.2f\t%.2f'%(mean, median))
        plt.show()

main()
