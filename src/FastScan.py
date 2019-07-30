"""
Methods and Classes invovled with object detection
on the Cornell Dataset using Faster RCNN
"""

import torch
import torchvision, argparse
from CornellDataset import CornellTranslator

class FastScanner:
    def __init__(self):
        """
        Model comes from pytorch torchvision, it is pretrained on mscoco and imagenet.
        """

        device = self.get_device()
        net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406] ,std=[0.229, 0.224, 0.225]
        )
        self.label_names = [ #luckly these are alread lower-case so no formatting needed
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        self.net = net.to(device)
        self.device = device
        self.normalize = normalize

    def get_device(self):
        """
        Utility function to find the right device for the data and deep learning model
        """

        if torch.cuda.is_available():
            device = torch.device('cuda:1')
        else:
            device = torch.device('cpu')

        return device

    def __call__(self, image):
        """
        Run inference on the input. Currently only supports a single image inference.
        Batches are undefined.

        image (numpy array): image to do inference on

        Returns: Dictionary with the following fields

            boxes: bounding boxes predicted on the image [x0, y0, x1, y1]
            labels: labels for each box
            scores: confidence for each box
        """

        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).float().to(self.device)
#        image = self.normalize(image)
        result = self.net([image])[0]
        bbox_areas, class_labels = [], []
        for bbox in result['boxes']:
            width = bbox[2].item() - bbox[0].item()
            height = bbox[3].item() - bbox[1].item()
            bbox_areas.append(width*height)
        for label in result['labels']:
            class_labels.append(self.label_names[label.item()])
        return {
            'boxes': result['boxes'],
            'labels': class_labels,
            'scores': result['scores'],
            'bbox_areas': bbox_areas,
        }

def main():
    """
    Perform a scan on the dataset looking for detections.
    Then output the results.
    Model = FasterRCNN
    """

    #parse the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to data folder')
    arguments = parser.parse_args()
    #build the dataset and model
    dataset = CornellTranslator(arguments.data_path, 'train')
    model = FastScanner()
    results, detection_count = [], 0
    for i in range(len(dataset)):
        image, _ = dataset[i]
        result = model(image)
#        results.append(result)
        if len(result['boxes']) > 0:
            detection_count += 1
        print(' %d/%d  '%(i+1, len(dataset)), end='\r')
    print('')
    print(detection_count)
