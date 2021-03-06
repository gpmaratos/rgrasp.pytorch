"""
Methods and Classes involved with the object detection on the
Cornell Dataset using yolov3 from darknet
"""

import os, cv2, argparse, math
import numpy as np
from CornellDataset import CornellTranslator
import matplotlib.pyplot as plt

class YoloScanner:
    def __init__(self, yolo_path, confidence=0.5, threshold=0.3):
        """
        Class which defines the yolo scanner, which does inference
        on a single image.

        yolo_path (string): path to the important yolo files, coco.names,
            yolov3.weights, yolov3.cfg

        confidence (float): defines how confident the scanner needs to be
            in a prediction for it to be considered a positive example

        threshold (float): threshold for non-maximum suppression
        """

        #load darknet
        with open(os.path.join(yolo_path, 'coco.names')) as f:
            label_names = f.read().strip().split('\n')
        weights_path = os.path.join(yolo_path, 'yolov3.weights')
        config_path = os.path.join(yolo_path, 'yolov3.cfg')
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        layer_names = net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #set some colors for if we are going to display our predictions
        colors = np.random.randint(0, 255, size=(len(label_names), 3), dtype='uint8')
        #set data members
        self.confidence = confidence
        self.threshold = threshold
        self.net = net
        self.layer_names = layer_names
        self.label_names = label_names
        self.colors = colors

    def __call__(self, image):
        """
        Run inference on the input. Results are given as a dictionary.

        image (numpy array): image to do inference on
        """
        #format numpy image
        shape = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255., shape, swapRB=True, crop=False)
        #run inference
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)
        #loop through the outputs generated by the model
        boxes, confidence_id, class_id, bbox_area, class_name = [], [], [], [], []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                Id = np.argmax(scores)
                confidence = scores[Id]
                #if confidence in the prediction is high enough treat it as a positive prediction
                if confidence > self.confidence:
                    box = detection[:4] * np.array((shape[1], shape[0], shape[1], shape[0]))
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width/2))
                    y = int(center_y - (height/2))
                    boxes.append([x, y, int(width), int(height)])
                    bbox_area.append(int(width) * int(height))
                    confidence_id.append(confidence.astype('float'))
                    class_id.append(Id)
                    class_name.append(self.label_names[Id])
        #run non-maximum supression and return resutls
        idxs = cv2.dnn.NMSBoxes(boxes, confidence_id, self.confidence, self.threshold)
        if len(idxs) > 0:
            self.display(idxs, boxes, class_id, confidence_id, image)
        return {
            'idxs': idxs,
            'boxes': boxes,
            'class_id': class_id,
            'confidence_id': confidence_id,
            'bbox_area': bbox_area,
            'class_name': class_name,
        }

    def display(self, idxs, boxes, class_id, confidence_id, image):
        """
        show image with box annotations on it.
        """

        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in self.colors[class_id[i]]]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            text = '%s: %.4f'%(self.label_names[class_id[i]], confidence_id[i])
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)

def display_area(results):
    """
    Show a histogram of the sqrt of areas of bounding boxes for examples
    that have atleast one positive example

    results (list): elements are results dictionaries and contain the inference from
        a single image
    """
    bbox_areas = []
    for result in results:
        bbox_area = result['bbox_area']
        if len(bbox_area) > 0:
            for bbox in bbox_area:
                bbox_areas.append(math.sqrt(bbox))
    plt.hist(bbox_areas)
    plt.show()

def display_class_predictions(results):
    class_predictions = {}
    for result in results:
        if len(result['class_name']) > 0:
            for class_name in result['class_name']:
                if class_name in class_predictions:
                    class_predictions[class_name] += 1
                else:
                    class_predictions[class_name] = 1
    class_names = class_predictions.keys()
    class_frequency = class_predictions.values()
    fig, ax = plt.subplots()
    ax.bar(class_names, class_frequency)
    ax.figure.autofmt_xdate()
    plt.show()

def main():
    """
    Perform a scan on the dataset looking for detections.
    Then output the results.
    Model = Yolov3
    """

    #parse the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('yolo_path', type=str, help='path to yolo directory')
    arguments = parser.parse_args()
    #build the dataset and model
    dataset = CornellTranslator(arguments.data_path, 'train')
    yolo_model = YoloScanner(arguments.yolo_path, confidence=0.50, threshold=0.3)
    #loop through the dataset recording what the model emits
    results, detection_count = [], 0
    for i in range(len(dataset)):
        image, _ = dataset[i]
        result = yolo_model(image)
        results.append(result)
        print(' %d/%d  '%(i+1, len(dataset)), end='\r')
    print('')
    display_class_predictions(results)
