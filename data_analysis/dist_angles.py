import math
import argparse
import matplotlib.pyplot as plt
from data_processing.dataset import CornellDataset

"""
Calculate the distribution of angles
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('DPATH', type=str,
        help='path to data folder')
    args = parser.parse_args()

    d_path = args.DPATH
    ds = CornellDataset(d_path, 'val')

    angs = []
    for bind, batch in enumerate(ds):
        print(' %d/%d '%(bind, len(ds)), end='\r')
        [angs.append(math.degrees(tup[2])) for example in batch\
            for tup in example[2]]
    plt.hist(angs)
    plt.show()
main()
