# **BME 595 DEEP LEARNING**

### **Soumendu Kumar Ghosh**


# **HOMEWORK-05:** *Neural Networks - CNN in PyTorch*

The results of different parts of the howework are described in this file.

##### Platform and Packages used:
- **Anaconda** 4.3.27, **Python** 3.6.2, **Pytorch** 0.2.0, **Torchvision** 0.1.9, **OpenCV** 3.1.0
- Processor: **Intel® Core™ i7Machine-6500U CPU @ 2.50GHz × 4 **, RAM: **15.4 GB**
- OS: **Ubuntu 16.04 LTS** 64 bit
###### NOTE: 
Till the last homework, Ubuntu 14.04 installed on Virtual Box was used for running all the codes. Since then, I have migrated to dual boot Ubuntu on the same machine which allowed me to use the full potential of my system resources (like CPU, RAM, etc.). This caused a substantial speed up of the running time of all the previous code. Please **DO NOT REFER** to submitted Homework 4 for comparison, it has been rerun on the new system and plots have been shown in this report only.

## **INTRODUCTION**
This assignment deals with training and testing of two types of datasets using the **Convolution NeuralNetwork API** of Pytorch's nn package.
- [**The MNIST Database**](http://yann.lecun.com/exdb/mnist/) of handwritten digits: The MNIST database has a training set of 60,000 examples, and a test set of 10,000 examples.
- [**The CIFAR-100 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html): The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 500 training images and 100 test images per class. These 100 classes are grouped in 20 superclasses. Each image is labeled with a coarse label for the *superclass* and fine label for *class* it belongs too.

## API
### Class: LeNet5
This is the superclass which defines the convolutional neural network model. This model comprises of:
- C1 - 1st **2D Convolution layer** (`input channel=1, output channel=6, kernel size=5, padding=2`): Performs 2D convolution on the input image . The padding is used so that our network can capture the features at the corner of the image as well. `ReLu` activation function has been used on the output of this layer.
- S2 - **2D MaxPool layer** (`kernel size=2`): Takes the maximum of each 2x2 region of the output of last layer. This is repeated over all the channels. This basically makes the network scale invariant. This function of this layer is subsampling.
- C3 - 2nd **2D Convolution layer**(`input channel=6, output channel=16, kernel size=5`): Performs 2D convolution on the outputs of the last layer  followed by `ReLu` activation function.
- S4 - **2D MaxPool layer** (`kernel size=5`): Performs subsampling.
- C5 - Fully connected layer (`input size=16*5*5,output size=120`): This is actually a 3rd **2D Convolution layer** (`kernel size=5`), where each unit is connected to a 5x5 neighbourhood on all 16 of previous layer output channels. As the size of 


### Class: img2num
  - The file *img2num.py* implements this class.
  - `__init__`: The class constructor performs the following tasks:
    - Initializes hyperparameters for training and validation
    - Downloads the *MNIST* dataset on the host PC using `torch.utils.data.DataLoader`.
    - Creates neural network model and MSE loss function using Pytorch's **nn** package.
    - Instantiates an SGD optimizer using **optim** package which will update the model parameters (weights) and predefined learning rate.
  - `train()`:
    - Initially the data and target labels are wrapped in Variable
    - Forward pass is performed on a mini-batch of the input data (viewed as 1D). MSE loss is calculated using the predefined loss function. Backward pass is performed using one-hot target labels. The predefined optimizer is used to update the theta(weights). This cycle is repeated for each batch of data in the training dataset.
    - The trained network is validated by feeding the validation data to the trained network model and comparing the predicted labels with the target labels. Loss and accuracy is computed.
    - This process of training and validation is repeated for predefined number of epochs.
  - `[int] forward([28x28 ByteTensor] img)`:
    - This method takes a single image from MNIST dataset and feeds it to the trained network. It returns the predicted label(class) of the input image.

### Hyperparameters for the API
|Type|MyImg2Num|NnImg2Num|
|----|---------|---------|
|Training Batch Size|60|60|
|Validation Batch Size|1000|1000|
|Network Structure|(784, 512, 256, 64, 10)|(784, 512, 256, 64, 10)|
|Learning Rate|0.1|15.0|
|Number of Epochs|30|30|
