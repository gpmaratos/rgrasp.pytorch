import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

"""
Script to determine if overlap exists in the anchors
the result is that using an achor of 16 pixels is best
"""

lst = []

def count_overlap(tup):
    count = 0
    temp = []
    for rec in tup:
        pos = (rec.x_pos, rec.y_pos)
        if pos in temp:
            count += 1
        else:
            temp.append(pos)
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds = CornellDataset(d_path, 'train')

    for bind, batch in enumerate(ds):
        print(' %d/%d '%(bind, len(ds)), end='\r')
        [lst.append(count_overlap(example[2])) for example in batch]

    plt.hist(lst)
    plt.show()
main()
