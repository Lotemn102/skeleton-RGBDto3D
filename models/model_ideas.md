# Models Ideas

## OpenPose (Supervised, single-framed)
### Network Architecture

Input (RGB, shape: (H, W, 3)) -> VGG19 -> 

<p align="center">
  <img width="600" height="" src="https://miro.medium.com/max/1094/1*FEMui63FL-znpL64lgf7Mw.png">
</p>

 Network output: [(heatmap, shape: (H, W, 19)), (19 vectors, shape: (H, W, 19 * 2))]

-> Inference -> Final output (19 points, 19 vectors)

Where:
- VGG19 - For features extraction.
- 2 sub-nets: The first one (top) extracts the keypoints, the second one (bottom) extracts the connections between the keypoints. After the first stage, both sub-net outputs concatenated with the original features. The concatenated output is inserted to a second stage of `t` layers. In the paper `t=6`.
- Greedy inference - For each detected point, they "connect" the point to each other point, creating a line. They then measure the confidence of the formed line using the the   predicted part affinity field (Described as `L` in the figure).
- The network outputs consists of 2 tensors: The first one is a (H, W, 19) tensor, each layers represents specific keypoint `j`. In each layer, the pixels that the network predicts that are part of keypoint `j`, will have higher value. The second one is a (H, W, 19 * 2) tensor. Each 2 successive layers represents a vector. The first layer of each pair represents the first point of the vector, and the second layer the second point.


### Loss
2 loss functions, one for each sub-net. Both are L2 loss. 
<p align="center">
  <img width="400" height="" src="https://miro.medium.com/max/840/1*dPFmjRbXDVnMYkiAvowyEA.png">
</p>

Where:
- `p` is a single pixel.
- `S(p)` is a 1d vector, consists of the confidence score for body part `j` at image location `p`.
- `L(p)` is 2d vector, consists of the directional vector for limb `c` at image location `p`.
- `W(p)` is a weighting function used to ignore samples with missing values. `W(p) = 0` when the annotation is missing at an image location `p`. 

Overall objective:
<p align="center">
  <img width="200" height="" src="https://miro.medium.com/max/392/1*NtVifzXCAQr3za3KAe8P3g.png">
</p>


### Links
- [Paper](https://arxiv.org/pdf/1812.08008.pdf)
- [Code](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [PyTorch implementation of the network](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) - They used MobileNet instead of VGG19 for the feature extraction.
- [Nice blog post about the architecture](https://medium.com/analytics-vidhya/understanding-openpose-with-code-reference-part-1-b515ba0bbc73)

### Required changes for our data
- Change the VGG19 input shape (?)
- Change number of keypoints in the part confidence map (S) to 39
- Change the keypoints representation to 3d
- Implement vicon datapoints convertor to COCO format

### Questions
- Do we want [A] to change the network s.t the output will be in 3d representation or [B] re-train the 2d network on our data, and reconstruct the 3d points from it's output? (that's what they did in [openpose-3d](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md#3-d-reconstruction), they used Levenberg–Marquardt algorithm for reconstruction, assuming 2 images as input from from stereo camera, also done in [here](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation-3d-demo.pytorch) using method described in [this paper](https://arxiv.org/pdf/1712.03453.pdf)).
- If [A], how? Some ideas:
  - Use a 3d matrix of (max_vicon_x, max_vicon_y, max_vicon_z). Scale all points before training s.t. each point is in a seperate pixel, re-scale the points back in inference. What layers can be used for the 2d->3d transformation?
- If [B], can we extract the 2 seperate images for each frame from realsense?
