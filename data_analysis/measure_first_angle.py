"""
Measure the first vector so that I can see if I can use it for measuring the orientation
"""

import math
import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

"""
Calculate the distribution of angles
"""
def extract_vector(tup_a, tup_b):
    diff_x = tup_a[0] - tup_b[0]
    diff_y = tup_a[1] - tup_b[1]
    dist = math.sqrt(diff_x**2 + diff_y**2)
    if diff_x < 0:
        diff_y *= -1
        ang = math.acos(diff_y/dist)
    else:
        ang = math.acos(diff_y/dist)
    return diff_x, diff_y, dist, math.degrees(ang)

def calc_first(tup):
    x1, y1, d1, a1 = extract_vector(tup[0], tup[1])
    x2, y2, d2, a2 = extract_vector(tup[0], tup[-1])
    if d1 < d2:
        return a2
    else:
        return a1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds = CornellDataset(d_path, 'train')

    lst = []
    for bind, batch in enumerate(ds):
        print(' %d/%d '%(bind, len(ds)), end='\r')
        [lst.append(calc_first(tup)) for example in batch\
            for tup in example[2]]
    plt.hist(lst)
    plt.show()
main()
