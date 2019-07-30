"""
@package docstring

Namespace for classes and methods that run analysis on data from the Cornell Dataset
Depends on CornellTranslator for many of the tasks
"""

import os, argparse, math
import numpy as np
from CornellDataset import CornellTranslator
from YoloScan import YoloScanner
from FastScan import FastScanner
from RecGrasp import RectangleGrasper
import matplotlib.pyplot as plt

class CornellAnalyzer:
    """
    Holds the path to the data and a translator object. Runs various methods on
    the contents of that directory.
    """

    def __init__(self, data_path, selected_set):
        """
        z.txt is parsed, and class names and image ids are extracted.

        data_path (string): path to the data

        selected_set (string): either train val or test, chooses the dataset split to use
        """

        #set some class members
        self.data_path = data_path
        self.dataset = CornellTranslator(data_path, selected_set)
        self.model = RectangleGrasper()
        #read class names and build mapping from image id to class name
        with open(os.path.join(self.data_path, 'z.txt')) as f:
            data = f.read().strip().split('\n')
        class_names, id_to_class, class_to_id = [], {}, {}
        #go through z.txt building mappings from id->class and inverse
        for line in data:
            line = line.split()
            image_id = int(line[0])
            if image_id in self.dataset.item_index:
                class_name = line[2].lower()
                if image_id not in id_to_class:
                    id_to_class[image_id] = class_name
                    if class_name not in class_to_id:
                            class_to_id[class_name] = [image_id]
                    else:
                        class_to_id[class_name].append(image_id)
        #extract classes present in the selected set
        for index in self.dataset.item_index:
            name = id_to_class[index]
            if name not in class_names:
                class_names.append(name)
        self.id_to_class = id_to_class
        self.class_to_id = class_to_id
        self.class_names = class_names

    def get_class_names(self):
        """
        Get the class names assigned in z.txt.
        Return the list of unique classes.
        """

        return self.class_names

    def find_class_overlaps(self, label_path):
        """
        Given a file of class names, compare them with the Cornell
        dataset to determine how well they overlap.

        label_path (string): path to label file, the labels are expected
            to be in a single column, one label per row. each name will
            be converted to pure lower case.
        """

        with open(label_path) as f:
            other_class_names = f.read().strip().split('\n')
        other_class_names = [name.lower() for name in other_class_names]
        class_name_intersection = []
        for name in self.class_names:
            if name in other_class_names:
                class_name_intersection.append(name)
        return class_name_intersection

    def count_rectangles(self):
        """
        Create a histogram of the number of rectangles per image for the whole
        dataset.
        """

        rectangles = []
        for i in range(len(self.dataset)):
            _, labels = self.dataset[i]
            rectangles.append(len(labels))
            print(" %d/%d  "%(i+1, len(self.dataset)), end='\r')
        print('')
        plt.hist(rectangles)
        plt.tight_layout()
        plt.show()

    def count_anchor_overlaps(self):
        """
        Count the number of overlaps in the labels. Display the result
        in a histogram.
        """

        overlaps = []
        for i in range(len(self.dataset)):
            _, labels = self.dataset[i]
            targets = self.model.create_targets([labels])[0]
            positions, overlap = [], 0
            for target in targets:
                if not target['position'] in positions:
                    positions.append(target['position'])
                else:
                    overlap += 1
            overlaps.append(overlap)
            print(" %d/%d  "%(i+1, len(self.dataset)), end='\r')
        print('')
        plt.hist(overlaps)
        plt.tight_layout()
        plt.show()

    def calculate_mean_pixel_value(self):
        """
        Calculates the mean pixel value for each image, and displays the distribution.
        This is mostly usefull for determining the quality of the background
        subtraction.
        """

        mean_distribution = []
        for i in range(len(self.dataset)):
            image, _ = self.dataset[i]
            mean_distribution.append(image.mean())

            print(" %d/%d  "%(i+1, len(self.dataset)), end='\r')
        print('')
        plt.hist(mean_distribution)
        plt.tight_layout()
        plt.show()

    def compute_rectangle_area(self):

        if self.model.backbone_type == 'vgg16':
            anchor_width = 32

        areas = []
        for i in range(len(self.dataset)):
            _, labels = self.dataset[i]
            targets = self.model.create_targets([labels])[0]
            for target in targets:
                width = math.pow(2, target['width'])*anchor_width
                length = math.pow(2, target['length'])*anchor_width
#                areas.append(math.sqrt(width*length))
                areas.append(length)
            print(" %d/%d  "%(i+1, len(self.dataset)), end='\r')
        print('')
        plt.hist(areas)
        plt.tight_layout()
        plt.show()

    def compute_class_frequency(self, class_overlaps):
        """
        Given a list of class names that exist in the Cornell Dataset
        find how frequent this subset of classes is in the dataset

        class_overlaps (list): elements are strings of names of labels
            in the dataset
        """

        count = 0
        for name in class_overlaps:
            count += len(self.class_to_id[name])
        return count/len(self.dataset)

    def scan_classes_yolo(self, yolo_path, class_names):
        """
        run the yolo scanner on all examples in the dataset with
        the class labels from class_names. count number of
        detections for each class. A detection here is defined
        as 1 if yolo finds anything and 0 else

        yolo_path (string): path to yolo directory

        class_names (list): elements are names of classes
        """

        scanner = YoloScanner(yolo_path)
        #for each class run yolo and create list of results
        results = []
        positions = []
        total_areas = []
        for i in range(len(class_names)):
            print(" Scanning Dataset by Class: %d/%d  "%(i+1, len(class_names)), end='\r')
            image_ids = self.class_to_id[class_names[i]]
            detection_count, areas = 0, []
            for image_id in image_ids:
                #find the image indicies in the cornell dataset using a reverse search
                index = int(np.where(self.dataset.item_index == image_id)[0])
                #now run the scanner and collect the results
                image, _ = self.dataset[index]
                result = scanner(image)
                if len(result['boxes']) > 0:
                    #boxes are list of lists
                    for box in result['boxes']:
                        x0, y0, x1, y1 = box
                        positions.append(( (x0+x1)/2 , (y0+y1)/2))
                    detection_count += 1
                    areas.append([math.sqrt(area) for area in result['bbox_area']])
                    for _area in result['bbox_area']: #using ugly leading underscore
                        total_areas.append(math.sqrt(_area))
            results.append({
                'class_name': class_names[i],
                'detection_count': detection_count/len(image_ids),
                'bbox_areas': areas,
            })
        print('')
        ind = [i for i in range(len(total_areas)) if total_areas[i] < 200]
        filtered_set = [positions[i] for i in ind]
        import pdb;pdb.set_trace()
        return results

    def scan_classes_faster_rcnn(self, class_names):
        """
        Scan the dataset using the faster rcnn model and get results for
        analysis. There is a lot of code duplication here with scan_classes_yolo
        which can be fixed later. Arguments are the same besides yolo_path.
        """

        scanner = FastScanner()
        #for each class, run Faster RCNN and create list of results
        results = []
        for i in range(len(class_names)):
            print(" Scanning Dataset by Class: %d/%d  "%(i+1, len(class_names)), end='\r')
            image_ids = self.class_to_id[class_names[i]]
            detection_count, areas = 0, []
            for image_id in image_ids:
                #find the image indicies in the cornell dataset using a reverse search
                index = int(np.where(self.dataset.item_index == image_id)[0])
                #now run the scanner and collect the results
                image, _ = self.dataset[index]
                result = scanner(image)
                if len(result['boxes']) > 0:
                    detection_count += 1
                    areas.append([math.sqrt(area) for area in result['bbox_areas']])
            results.append({
                'class_name': class_names[i],
                'detection_count': detection_count/len(image_ids),
                'bbox_areas': areas,
            })
        print('')
        return results

def main():
    """
    Run Tests Here
    If you pass a file with class labels, and either the yolo or rcnn flags, then
    this function will compute metrics for the scanners on the overlapping datasets.
    """

    #parse the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('selected_set', type=str, help='dataset to use')
    parser.add_argument('-f', type=str, help='optional argument for class analysis')
    parser.add_argument('-y', type=str, help='pass yolo directory for performing yolo scan')
    arguments = parser.parse_args()

    analyzer = CornellAnalyzer(arguments.data_path, arguments.selected_set)
    if arguments.y:
        analyzer.scan_classes_yolo(arguments.y, list(analyzer.class_to_id.keys()))

main()
