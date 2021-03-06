"""
@package docstring

Namespace for the grasp detection module. Given an image, or a portion of an image because
it is meant to be part of a system, find a suitable grasping location and predict
the configuration of an end effector (parallel gripper plates).
"""

import argparse, torch, numpy as np, torchvision, itertools, random, math, datetime, os
from torch import nn
from sklearn.metrics import precision_recall_fscore_support
from CornellDataset import CornellTranslator
import logging

def build_vgg16_headless():
    """
    returns the pretrained vgg16 convolutional model trained on imagenet by pytorch
    """

    return torchvision.models.vgg16_bn(pretrained=True).features

def build_simple_detector(backbone_type):
    """
    returns a simple linear regressor that accepts X image features and predicts
    two values for each anchor, scores that indicate the possibility of either having
    or not having a grasp for the current anchor. X depends on what the backbone is

    backbone_type (string): supports "vgg16", uses this to determine how many input
        features the network accepts
    """

    if backbone_type == 'vgg16':
        return nn.Conv2d(512, 2, (1, 1))

def build_simple_regressor(backbone_type):
    """
    returns a simple linear regressor that accepts X image features and predicts
    four values for each anchor, scores that indicate the possibility of either having
    or not having a grasp for the current anchor. X depends on what the backbone is.
    The four values are offsets of (x_center, y_center, width, height, and angle), which
    is how the model predicts the position of the parallel gripper plates

    backbone_type (string): supports "vgg16", uses this to determine how many input
        features the network accepts
    """

    if backbone_type == 'vgg16':
        return nn.Conv2d(512, 5, (1, 1))

def build_simple_head(backbone_type):
    """
    Combination of build_simple_regressor and build_simple_detector
    """

    if backbone_type == 'vgg16':
        return nn.Conv2d(512, 7, (1, 1))


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    return device

class RectangleGrasper(nn.Module):
    """
    Model which predicts where grasps lie in an image, and predicts a rectangle.
    Visual features are calculated using a pretrained network, trained on imagenet.
    """

    def __init__(self):
        """
        Initializes the components of the network, it has a backbone to extract
        visual features, and two network outputs. A detector and regressor.
        """

        super(RectangleGrasper, self).__init__()
        self.backbone = build_vgg16_headless()
        self.backbone_type = 'vgg16'
        #build the heads
        self.grasp_head = build_simple_head('vgg16')

    def forward(self, images, labels):
        """
        Forward pass of the RectangleGrasper model. What it returns depends on its mode.
        If it is in evaluation mode, then it returns predictions <what those look like
        are to be defined later>. If it is in training mode,
        it retuns the total loss for calculating the gradient of the weights
        and running backpropogation. If it is in evaluation mode then the labels are
        ignored.

        images (list): elements are numpy arrays of shape (H x W x C)

        labels (list): elements are lists of rectangles (which are lists of 4 tuples, each
            tuple is a coordinate)
        """

        #create the tensor and calculate the visual features
        tensor = self.convert_to_pytorch(images)
        visual_features = self.backbone(tensor)
        #get output from head layers
        detections = self.grasp_head(visual_features)
        #if training then calculate the loss and return else return
        if self.training:
            targets = self.create_targets(labels)
            return self.compute_loss(targets, detections)
        else:
            return detections

    def extract_vector(self, tup_a, tup_b):
        """
        Helper function for translate_rectangle, calculates the longest edge of
        the rectangle and then returns the width, length, and angle based off that.

        tup_a, tup_b (tuple): elements are floats and represent coordinates
        """

        diff_x = tup_a[0] - tup_b[0]
        diff_y = tup_a[1] - tup_b[1]
        dist = math.sqrt(diff_x**2 + diff_y**2)
        if diff_x < 0:
            diff_y *= -1
            ang = math.acos(diff_y/dist)
        else:
            ang = math.acos(diff_y/dist)
        return dist, ang

    def translate_rectangle(self, rectangle):
        """
        Helper function for create_targets. Converts a rectangle given as coordinates
        of the corners, to a tuple containing offsets from anchor boxes, length, width,
        and angle.

        rectangle (tuple): rectangle defined by coordinates, which are floats of x, y
        """
        x_center = sum([coord[0] for coord in rectangle])/4
        y_center = sum([coord[1] for coord in rectangle])/4
        dist1, ang1 = self.extract_vector(rectangle[0], rectangle[1])
        dist2, ang2 = self.extract_vector(rectangle[0], rectangle[-1])
        if dist1 < dist2:
            width = dist1
            length = dist2
            angle = ang2
        else:
            width = dist2
            length = dist1
            angle = ang1
        return x_center, y_center, width, length, angle

    def create_targets(self, labels):
        """
        Format the labels so that the loss can be calculated. What is returned is a
        list of dictionaries that is used by the loss functions.

        labels (list): elements are lists of rectangles

        targets (list): elements are lists of dictionaries containing the encoded info
        """

        if self.backbone_type == 'vgg16':
            #there are 10 anchors each 32 bits
            anchor_width = 32
            anchor_count = 10
        targets = []
        for label in labels:
            positions = []
            for rectangle in label:
                #get the position of the rectangle and its area and angle
                x_center, y_center, width, length, angle = self.translate_rectangle(rectangle)
                #encode the targets
                x_position = int(x_center / anchor_width)
                y_position = int(y_center / anchor_width)
                if x_position >= anchor_count:
                    x_position = anchor_count -1
                if y_position >= anchor_count:
                    y_position = anchor_count -1
                x_offset = (x_center % anchor_width)/32
                y_offset = (y_center % anchor_width)/32
                width = math.log(width/anchor_width)/math.log(2)
                length = math.log(length/anchor_width)/math.log(2)
                angle /= 6.28 #normalize by approx 2 pi
                positions.append({
                    'position': (x_position, y_position),
                    'offsets': (x_offset, y_offset),
                    'width': width,
                    'length': length,
                    'angle': angle,
                })
            targets.append(positions)
        return targets

    def compute_loss(self, targets, detections):
        """
        Computes the loss. The negative class is much larger than the positive
        so the negative class is sampled based off of the confidence in false positives.
        """

        if self.backbone_type == 'vgg16':
            anchor_number = 10
        device = next(self.parameters()).device
        weight = torch.tensor([1., 1.5]).to(device)

        inp, targ = [], []
        input_regression, target_regression = [], []
        for i in range(len(targets)):
            #set some local variables
            target = targets[i]
            detection = detections[i, :2]
            regression = detections[i, 2:]
            #extract positive indices and set negative size for sampling
            positive_indices = [rectangle['position'] for rectangle in target]
            negative_size = len(positive_indices)*3
            #find best false positives
            differences = []
            for p_x, p_y in itertools.product(range(anchor_number), range(anchor_number)):
                if detection[0, p_x, p_y] < detection[1, p_x, p_y]:
                    diff = detection[1, p_x, p_y] - detection[0, p_x, p_y]
                    differences.append((diff, p_x, p_y))
            sorted_differences = sorted(differences, key=lambda z:z[0], reverse=True)
            neg_inds = []
            j = 0
            if len(sorted_differences) > 0:
                while j < negative_size:
                    if not sorted_differences[j][1:] in positive_indices:
                        neg_inds.append(sorted_differences[j][1:])
                    j += 1
                    if j > len(sorted_differences)-1:
                        break
            if len(neg_inds) < negative_size:
                while len(neg_inds) < negative_size:
                    pair = (random.randint(0, detections.shape[2]-1),
                        random.randint(0, detections.shape[3]-1))
                    if not pair in positive_indices:
                        neg_inds.append(pair)
            #build array
            for rectangle in target:
                target_regression.append(torch.tensor([
                    rectangle['position'][0],
                    rectangle['position'][1],
                    rectangle['width'],
                    rectangle['length'],
                    rectangle['angle']
                ]))
            for pair in positive_indices:
                input_regression.append(regression[:, pair[0], pair[1]])
                inp.append(detection[:, pair[0], pair[1]])
                targ.append(1)
            for pair in neg_inds:
                inp.append(detection[:, pair[0], pair[1]])
                targ.append(0)
        #compute detection loss
        input_tensor = torch.stack(inp)
        target_tensor = torch.tensor(targ).to(device)
        detection_loss = torch.nn.functional.cross_entropy(input_tensor, target_tensor,
            weight=weight)
        inps = np.array([0 if t[0] > t[1] else 1 for t in inp])
        targs = target_tensor.cpu().numpy()
        prec, rec, f1, sup = precision_recall_fscore_support(
            targs,
            inps,
            average='binary'
        )
        #compute the regression loss
        input_regression_tensor = torch.stack(input_regression)
        target_regression_tensor = torch.stack(target_regression).to(device)
        regression_loss = torch.nn.functional.smooth_l1_loss(
            input_regression_tensor,
            target_regression_tensor,
            reduction='none',
        ).mean(0)

        return {
            'loss': 2*detection_loss + regression_loss,
            'detection_loss': detection_loss,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'regression_loss': regression_loss.mean(),
            'expanded_regression':regression_loss
        }

    def convert_to_pytorch(self, images):
        """
        Transform the image into a pytorch tensor for training, and return the tensor.
        tensor will be placed on the same device as the calling object, because it is
        expected that the tensor will be part of some calculation involving the parameters
        of this object.

        images (list): elements are numpy arrays of shape (H x W x C)
        """

        #device is infered from the device of the first set of parameters
        device = next(self.parameters()).device
        tensor = torch.tensor(images, dtype=torch.float).to(device) / 255
        return tensor.permute(0, 3, 1, 2)

def record(msg):
    """
    Logging function used during training. Simply prints to the terminal
    and logs message in INFO.
    """

    print(msg)
    logging.info(msg)

def fit(model, dataset, weight_path, checkpoint=0):
    """
    Fit parameters to the dataset. Runs 500 epochs of stochastic
    gradient decent. Logs the training results in a file, and outputs
    them to the terminal. Dataset is processed in batch sizes of 20.
    The value of checkpoint determines how many epochs must pass.
    Set checkpoint to 0 to never save intermediate weights.
    before the weights are saved, the weights of this network are
    approximately 57 MB. In the future it would be nice to have validation
    results while training.
    """

    #log in file
    logging.basicConfig(filename=os.path.join(weight_path, 'train.log'), level=logging.INFO)
    #create optimizer
    opt = torch.optim.SGD([
        {'params': model.parameters()}
        ],lr=1e-4, momentum=0.9, weight_decay=1e-4
    )
    #find device
    device = next(model.parameters()).device
    #run training loop
    for i in range(1, 501):
        record("Epoch %d"%(i))
        for images, labels, percentage in dataset.yield_batches(20):
            #read and normalize the batch of images
            result = model(images, labels)
            #run backpropogation and update the parameters
            opt.zero_grad()
            result['loss'].backward()
            opt.step()
            msg = 'Completed %.2f ~ DLoss %.5f ~ RLoss %.5f ~ Precision %.5f ~ Recall %.5f ~ F1 %.5f ~ X ~ %.5f ~ Y ~ %.5f ~ W ~ %.5f ~ L ~ %.5f ~ Theta ~ %.5f'%(
                percentage,
                result['detection_loss'].item(),
                result['regression_loss'].item(),
                result['precision'],
                result['recall'],
                result['f1_score'],
                result['expanded_regression'][0].item(),
                result['expanded_regression'][1].item(),
                result['expanded_regression'][2].item(),
                result['expanded_regression'][3].item(),
                result['expanded_regression'][4].item(),
            )
            record(msg)
        if checkpoint > 0 and i % checkpoint == 0:
            model = model.to(torch.device('cpu'))
            m_pref = 'model-epoch-%d-'%(i) + str(datetime.date.today())
            w_path = os.path.join('vgg16', m_pref+'.pt')
            record("\nSaving Model at: %s"%(w_path))
            torch.save(model.state_dict(), w_path)
            model = model.to(device)

    print('Training Concluded, saving final weights.')
    model = model.to(torch.device('cpu'))
    m_pref = 'model-final-'%(i) + str(datetime.date.today())
    w_path = os.path.join('vgg16', m_pref+'.pt')
    record("\nSaving Model at: %s"%(w_path))
    torch.save(model.state_dict(), w_path)
    model = model.to(device)
    return model

def calculate_all_results(data_path, weight_path):
    """
    Calculates all the metrics for all of the datasets
    """

    #get device
    device = get_device()
    #create model
    model = RectangleGrasper().to(device)
    #log in file
    logging.basicConfig(filename=os.path.join(weight_path, 'metrics.log'), level=logging.INFO)
    #create the datasets
    train_dataset = CornellTranslator(data_path, 'train')
    val_dataset = CornellTranslator(data_path, 'val')
    test_dataset = CornellTranslator(data_path, 'test')
    #extract the names of the weights files
    files = os.listdir(weight_path)
    weight_files = []
    for file in files:
        _, extension = file.split('.')
        if extension == 'pt':
            weight_files.append(file)
    ordered_files = []
    for file in weight_files:
        epoch = int(file.split('-')[2])
        ordered_files.append((epoch, file))
    ordered_files = sorted(ordered_files, key=lambda x:x[0])
    #for each weight file, process its performance on all datasets
    for ordered_file in ordered_files:
        weights_path = os.path.join(weight_path, ordered_file[1])
        #load model
        model.load_state_dict(torch.load(weights_path))
        #evaluate on datasets
        train_results = eval(model, train_dataset)
        val_results = eval(model, val_dataset)
        test_results = eval(model, test_dataset)
        #record results
        record("Train: " + train_results)
        record("Val: " + val_results)
        record("Test: " + test_results)

def eval(model, dataset):
    """
    Evaluates the performance of the model on the selected dataset.
    Returns the results as a string, because I only used this to print
    right away.
    """

#    model = model.eval()
    detection_losses, regression_losses, precisions, recalls, f1_scores = [], [], [], [], []
    x, y, w , l, theta = [], [], [], [], []
    for images, labels, percentage in dataset.yield_batches(20):
        #read and normalize the batch of images
        result = model(images, labels)
        detection_losses.append(result['detection_loss'].item())
        regression_losses.append(result['regression_loss'].item())
        precisions.append(result['precision'])
        recalls.append(result['recall'])
        f1_scores.append(result['f1_score'])
        x.append(result['expanded_regression'][0].item())
        y.append(result['expanded_regression'][1].item())
        w.append(result['expanded_regression'][2].item())
        l.append(result['expanded_regression'][3].item())
        theta.append(result['expanded_regression'][4].item())
        print(' Progress %.2f  '%(percentage), end='\r')
    print('')
    msg = '~ DLoss ~ %.5f ~ RLoss ~ %.5f ~ Precision ~ %.5f ~ Recall ~ %.5f ~ F1 %.5f ~ X ~ %.5f ~ Y ~ %.5f ~ W ~ %.5f ~ L ~ %.5f ~ Theta ~ %.5f'%(
        sum(detection_losses)/len(detection_losses),
        sum(regression_losses)/len(regression_losses),
        sum(precisions)/len(precisions),
        sum(recalls)/len(recalls),
        sum(f1_scores)/len(f1_scores),
        sum(x)/len(x),
        sum(y)/len(y),
        sum(w)/len(w),
        sum(l)/len(l),
        sum(theta)/len(theta)
    )
    return msg

def main():
    """
    Train the grasping model and save the weights in a folder specified on
    the command line
    """

    #set up logger
    #parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('weights_path', type=str, help='path to weights')
    arguments = parser.parse_args()
    calculate_all_results(arguments.data_path, arguments.weights_path)
#    #create dataset
#    dataset = CornellTranslator(arguments.data_path, 'test')
#    #get gpu if exists
#    device = get_device()
#    #create model
#    model = RectangleGrasper().to(device)
#    model.load_state_dict(torch.load(arguments.weights_path))
#    #run eval
#    eval(model, dataset)

main()

#def eval():
#    """evaluation mode"""
#    #parse argument
#    parser = argparse.ArgumentParser()
#    parser.add_argument('data_path', type=str, help='path to data folder')
#    parser.add_argument('weights_path', type=str, help='path to weights')
#    arguments = parser.parse_args()
#    #create dataset
#    dataset = CornellTranslator(arguments.data_path, 'train')
#    #create model
#    model = RectangleGrasper()
#    model.load_state_dict(torch.load(arguments.weights_path))
#    model = model.eval()
#
#    for images, labels, percentage in dataset.yield_batches(20):
#        #read and normalize the batch of images
#        result = model(images, labels)
#        for i, j in itertools.product(range(10), range(10)):
#            if result[0, 0, i, j] < result[0, 1, i, j]:
#                offsets = result[0, 2:, i, j]
#                x_pred = i*32 + offsets[0].item()
#                y_pred = j*32 + offsets[1].item()
#                w_pred = math.pow(2, offsets[2].item())*32
#                l_pred = math.pow(2, offsets[3].item())*32
#                t_pred = offsets[4].item() * 6.28
#                import pdb;pdb.set_trace()
#                print('Prediction at %d, %d'%(i, j))
