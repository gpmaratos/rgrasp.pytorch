"""@package docstring
Methods and classes involved with analysing the
quality of the background subtraction. The quality
of the measure is determined by the mean pixel value
in an image. The idea being a lower average value is a
better subtraction because most pixels will be empty.
"""

import numpy as np
import argparse
from CornellDataset import CornellTranslator
import matplotlib.pyplot as plt

def calculate_metrics(mean_pixel_dist):
    """
    Calculates a series of metrics

    mean_pixel_dist (list): list of mean pixel values for each image
    """
    numpy_mpd = np.array(mean_pixel_dist)
    average = np.mean(numpy_mpd)
    median = np.median(numpy_mpd)
    std = np.std(numpy_mpd)
    p25 = np.percentile(numpy_mpd, 25)
    p65 = np.percentile(numpy_mpd, 65)
    return {
        'average': average,
        'median': median,
        'std': std,
        'p25': p25,
        'p65': p65,
    }

def main():
    """
    Parses Command Line arguments. Passes through the dataset,
    calculates the average image pixel value, and prints metrics
    and a histogram of the data.

    data_path (string): path to datafolder

    selected_set (string): either train or val, selects the dataset to use
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('selected_set', type=str, help='either train or val')
    arguments = parser.parse_args()
    dataset = CornellTranslator(arguments.data_path, selected_set=arguments.selected_set)
    mean_pixel_distribution = []
    for i in range(len(dataset)):
        image, _ = dataset[i]
        mean_pixel_distribution.append(image.mean())
        print(' %d/%d  '%(i+1, len(dataset)), end='\r')
    print('')
    metrics = calculate_metrics(mean_pixel_distribution)
    print("pdist\tmean\tmedian\tstd\t25% \t65% ")
    formatted_metrics = "\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(
        metrics['average'],
        metrics['median'],
        metrics['std'],
        metrics['p25'],
        metrics['p65']
    )
    print(formatted_metrics)
    plt.hist(mean_pixel_distribution)
    plt.tight_layout()
    plt.show()

main()
