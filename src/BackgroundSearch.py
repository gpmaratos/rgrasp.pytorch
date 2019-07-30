"""
@package docstring
Methods for finding the optimal background. I make the assumption
that the lowest mean pixel value indicates the best
"""

import argparse, os, cv2

class BackgroundGenerator:
    """
    class that takes a data folder, and generates a background mapping.
    It assumes the background image names, and places the mapping file
    inside the same directory as the data
    """

    def __init__(self, data_path):
        """
        data_path (string): path to data
        """

        index = list(range(100, 950)) + list(range(1000, 1035))
        del index[32]; del index[64]

        self.index = ['pcd%04d'%ind + 'r.png' for ind in index]
        self.data_path = data_path
        self.named_list = [
            'pcdb0002r.png',
            'pcdb0003r.png',
            'pcdb0004r.png',
            'pcdb0005r.png',
            'pcdb0006r.png',
            'pcdb0007r.png',
            'pcdb0008r.png',
            'pcdb0010r.png',
            'pcdb0011r.png',
            'pcdb0012r.png',
            'pcdb0013r.png'
        ]

    def generate_background_map(self):
        """
        Iterates through the images in the dataset, and finds the
        best background, by subtracting the backgrounds from the
        image and comparing mean pixel values
        """
        with open(os.path.join(self.data_path, 'backgrounds.txt'), 'w') as f:
            for i in range(len(self.index)):
                best_background  = self.find_best_background(self.index[i])
                f.write("%s ~ %s\n"%(self.index[i], best_background))
                print(' %d/%d  '%(i+1, len(self.index)), end='\r')

    def find_best_background(self, image_name):
        """
        Checks each background subtraction, and chooses the image that
        gives the best.

        image_name (string): name of the image as it appears in the Dataset
        """

        image_array = cv2.imread(os.path.join(self.data_path, image_name))
        best_mean = 255.
        for name in self.named_list:
            background_image = cv2.imread(os.path.join(self.data_path, name))
            subtracted_image = image_array - background_image
            mean = subtracted_image.mean()
            if mean < best_mean:
                best_name = name
                best_mean = mean
        return best_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to data folder')
    arguments = parser.parse_args()
    BackgroundGenerator(arguments.data_path).generate_background_map()
    return 0

main()
