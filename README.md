# MaskKeypointRCNN-pytorch
pytorch implementation of mask keypoint R-CNN, which is able to detected bbox, keypoints and segmentation.

The example of dataset (which has 10 images of pedestrians labeled with keypoints and segmentation) is labeled by [via](@https://www.robots.ox.ac.uk/~vgg/software/via/). 
The dataset class format is consistent to pytorch's.

In training, if the loss doesn't converge to an ideal point, tuning the weight of mask loss and keypoint loss might help.
The weight given in the script is suitable for the example dataset. 
