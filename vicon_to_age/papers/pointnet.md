### PointNet Architecture
In the paper the authors described that the main properties of the problem of classifying point cloud data are:
- Point cloud is a set of points without specific order. So if for example, I have a set on N 3d points that is classified as a cube, each permutation of these N 3d points should result in the same classification.
- Point cloud is invariance under transformations. If I rotate the cube in space, it's classification remains the same.

To solve this issues, they have designed the following network:
<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/35609587/135868713-00d221e7-774e-4d01-9a75-25351798d99f.PNG">
</p>


This network solves each of the problems described above:
- A function that is invariant to permutations is a symmetric function, for example + or *. The authers showed that if 'h' and 'p' are continuous, and 'g' is symmetric, then: ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/35609587/135869088-4fcafd8d-937c-4dcf-810b-59584154964c.gif) 
is symmetric. In the network, they have choosen max pooling as a symmetric function, and for the continuous functions they chose MLP with 1-dimentional convolutional layer, followed by batch normalization and ReLU.
- In order to make the point cloud invariance under transformations, they have transformed the point cloud input into a canonical representation. They have used a smaller net for that (T-Net), which is a smaller version of the entire network. The input is transformed using this T-Net in training and in testing.
