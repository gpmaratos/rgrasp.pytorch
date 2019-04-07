import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

"""
Script to determine if overlap exists in the anchors
the result is that using an achor of 16 pixels is best
"""

lst = []

def extract_member(tup):
    return tup.theta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds = CornellDataset(d_path, 'train')

    for bind, batch in enumerate(ds):
        print(' %d/%d '%(bind, len(ds)), end='\r')
        [lst.append(extract_member(tup)) for example in batch\
            for tup in example[2]]

    plt.hist(lst)
    plt.show()
main()

