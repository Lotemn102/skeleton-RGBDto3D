# Models Ideas

## OpenPose (Supervised, single-framed)
### Network Architecture

Input (RGB) -> VGG19 -> 

<p align="center">
  <img width="600" height="" src="https://miro.medium.com/max/1094/1*FEMui63FL-znpL64lgf7Mw.png">
</p>

-> Greedy inference -> Output (19 2d points, 19 vectors)

Where:
- VGG19 - For features extraction.
- 2 sub-nets: The first one (top) extracts the keypoints, the second one (bottom) extracts the connections between the keypoints. After the first stage, both sub-net outputs concatenated with the original features. The concatenated output is inserted to a second stage of `t` layers. In the paper `t=6`.
- Greedy inference - For each detected point, they "connect" the point to each other point, creating a line. They then measure the confidence of the formed line using the the   predicted part affinity field (Described as `L` in the figure).


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

### Required changes in the network for our data
- Change the VGG19 input shape
- Change number of keypoints in the part confidence map (S) to 39
- Change vector dimensions in the part affinity field (L) to 3d
