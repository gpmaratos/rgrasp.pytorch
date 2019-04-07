"""
Measure the first vector so that I can see if I can use it for measuring the orientation
now it calculates rectangle length width and angle
"""

import math
import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

"""
Calculate the distribution of angles
"""
lst = []

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

def calc_tup(tup):
    x_cent = sum([coord[0] for coord in tup])
    y_cent = sum([coord[1] for coord in tup])
    x1, y1, d1, a1 = extract_vector(tup[0], tup[1])
    x2, y2, d2, a2 = extract_vector(tup[0], tup[-1])
    if d1 < d2:
        return d2
    else:
        return d1

def calc_coord(tup):
    [lst.append(coord[1]) for coord in tup]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds = CornellDataset(d_path, 'val')

    for bind, batch in enumerate(ds):
        print(' %d/%d '%(bind, len(ds)), end='\r')
        [calc_coord(tup) for example in batch\
            for tup in example[2]]
    plt.hist(lst)
    plt.show()
main()
