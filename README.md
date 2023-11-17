# Computer Vision Classification Models

This  contains popular computer vision classification models，which is mainly aimed at scientific research developers. The code is relatively simple but comprehensive.These models are widely used in the field of computer vision for tasks like image classification, object detection, and image recognition.
## Models Included

1. LeNet-5: A classic convolutional neural network model proposed by Yann LeCun in 1998 for handwritten digit recognition.

2. AlexNet: A deep convolutional neural network model proposed by Alex Krizhevsky et al. in 2012, which achieved significant performance improvements in the ImageNet image classification challenge.

3. VGGNet: A deep convolutional neural network model developed by the Visual Geometry Group (VGG) at the University of Oxford in 2014. It is known for its deep architecture with small convolutional filters.

4. GoogLeNet: A convolutional neural network model proposed by the Google Research team in 2014. It introduced the concept of Inception modules to efficiently reduce the number of parameters in the network.

5. ResNet: A residual neural network model developed by Microsoft Research in 2015. ResNet addresses the vanishing gradient problem in deep networks by introducing shortcut connections and bottleneck structures.

6. InceptionNet: A convolutional neural network model introduced by the Google Research team in 2014. It incorporates multiple scales of convolutions and uses 1x1 convolutions within the Inception module to reduce computational complexity.

7. DenseNet: A densely connected convolutional neural network model proposed by Gao Huang et al. in 2016. DenseNet enables direct connections between all layers, facilitating feature propagation and reuse.

## Usage

Each model in this repository is implemented as a separate Python script. You can run any of the scripts to train or test the respective model on your dataset. Please make sure that you have installed the required dependencies, including TensorFlow or PyTorch, before running the scripts.

To use a specific model, follow the instructions provided in the respective script's comments. You may need to adjust hyperparameters, dataset paths, and other settings according to your requirements.

Please note that the models provided here are for educational and research purposes. It is recommended to refer to the original research papers for a detailed understanding of the models and to use pre-trained models whenever possible to achieve better performance.

## License

This repository is licensed under the [MIT License](LICENSE). Feel free to use and modify the code according to your needs. However, please acknowledge the original authors and research papers when using these models.

## Contributions

Contributions to this repository are welcome. If you have implemented any additional computer vision classification models or have suggestions for improvements, please feel free to open a pull request.

## References

- [LeNet-5: Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/lenet/)
- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)](https://arxiv.org/abs/1409.1556)
- [Going Deeper with Convolutions (GoogLeNet)](https://arxiv.org/abs/1409.4842)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [Rethinking the Inception Architecture for Computer Vision (InceptionNet)](https://arxiv.org/abs/1512.00567)
- [Densely Connected Convolutional Networks (DenseNet)](https://arxiv.org/abs/1608.06993)