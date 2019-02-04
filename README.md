# pytorch.rgrasp

This is my work relating to applying the network found in
A Real-time Robotic Grasp Approach with Oriented Anchor Box
https://arxiv.org/abs/1809.03873

The code was designed to have a similar structure to the
facebook mask rcnn implementation,
https://github.com/facebookresearch/maskrcnn-benchmark

The network is designed to learn how to detect grasps from
the, very tiny, Cornell Grasping Datset. Because of this
I rely on pre-training, and carefully selected learning
rates to mitigate this challenge. It currently trains,
and will use a GPU if available.

Current list of Dependencies:

python 3.6 or greater

torchvision (for the pretrained model)

pytorch

scikit-image

numpy
