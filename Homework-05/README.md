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

# MNIST
## API
#### MODEL: LeNet5
This is the superclass which defines the convolutional neural network model. This model comprises of:
- C1 - 1st **2D Convolution layer** (`input channel=1, output channel=6, kernel size=5, padding=2`): Performs 2D convolution on the input image . The padding is used so that our network can capture the features at the corner of the image as well. `ReLu` activation function has been used on the output of this layer.
- S2 - **2D MaxPool layer** (`kernel size=2`): Takes the maximum of each 2x2 region of the output of last layer. This is repeated over all the channels. This basically makes the network scale invariant. This function of this layer is subsampling.
- C3 - 2nd **2D Convolution layer**(`input channel=6, output channel=16, kernel size=5`): Performs 2D convolution on the outputs of the last layer  followed by `ReLu` activation function.
- S4 - **2D MaxPool layer** (`kernel size=2`): Performs subsampling.
- C5 - **Fully connected layer** (`no of neurons=120`): This is actually a 3rd **2D Convolution layer** (`input size=16, output channel=120, kernel size=1`), where each unit is connected to a 5x5 neighbourhood on all 16 of previous layer output channels. As the size of the kernel is 1x1, this boils down to being fully connected with the previous layer. `ReLu` is the activation function.
- L6 - **Fully connected layer** (`no of neurons=84`): This is fully connected with C5.`ReLu` is the activation function.
- **OUTPUT LAYER** (`no of neurons=10`): The no of nodes here is same as the no of classes in MNIST dataset.
#### LOSS FUNCTION: MSE (Mean Square Loss)
#### OPTIMIZER: SGD (Stochastic Gradient Descent)

### Hyperparameters
|     Type            |  Value  |
|---------------------|---------|
|Training Batch Size  |   60    | 
|Validation Batch Size| 1000    |
|Learning Rate        |  1.0    |
|Number of Epochs     |   50    |


### Class: img2num
  - The file *img2num.py* implements this class.
  - `__init__`: The class constructor performs the following tasks:
    - Instantiates model, loss, optimizer, hyperparameters for training and validation.
    - Downloads the *MNIST* dataset on the host PC using `torch.utils.data.DataLoader`.
    - Loads a previously saved checkpoint (if present) consisting of last saved epoch, best accuracy, model state dictionary, optimizer dictionary, and list of results (training loss, validation loss, validation accuracy, and computation time for training). If no saved checkpoint is found, initializes list of results.
  - `train()`:
    - This method trains the model using training dataset and test the model usign validation dataset for predefined number of epochs. It also calculates result parameters like loss, accuracy, etc.
  - `[int] forward([28x28 ByteTensor] img)`:
    - This method takes a single image from MNIST dataset, unsqueezes it to size `1x1x28x28` and feeds it to the trained network. It returns the predicted label(class) of the input image.
  - At the end of each epoch, the epoch num, best accuracy, model state dictionary, optimizer dictionary, and list of results (training loss, validation loss, validation accuracy, and computation time for training) are saved, so that training can be resumed later from a saved checkpoint.

## **Results and Observations**
### **LeNet5 model (img2num) vs Basic (Fully connected) model (NnImg2Num)**
#### Loss vs Epoch Comparison
![Plot1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-05/MNIST/Results/Comparison_LossvsEpoch.png)
The LeNet5 model reaches a validation loss at epoch no 4 lesser than what is achiveved by basic model at epoch no 50. From the plot, it can be observed, that the LeNet5 model converges at epoch no 10 while the basic model converges at epoch no 20.

#### Time vs Epoch Comparison
![Plot2](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-05/MNIST/Results/Comparison_TimevsEpoch.png)
It can be observed from the plot the training in each epoch takes more time in LeNet5 model than in the other. This is evident from the fact that the LeNet5 model is complex with more no of parameters, hence takes more time to train. However, from the table below we can observe that it takes less time to converge for LeNet5 as it reaches very high validation accuracy at less no of epochs. If we consider the whole training and validation phase for 50 epochs, then total time is obviously larger for LeNet5.

#### Accuracy Comparison
![Plot3](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-05/MNIST/Results/Comparison_ValidationAccuracy.png)
LeNet5 model obtains better accuracy than Basic Model.

#### **Table**: Comparison of Results of LeNet5 and Basic Model

|  Model     |   Epoch   | Training Loss| Validation Loss| Total Time(s)| Validation Accuracy|
|------------|-----------|--------------|----------------|--------------|--------------------|
|LeNet5 Model|     10    |    0.001640  |   0.001844     |      103.7   |     99.23%         |
|LeNet5 Model|     50    |    0.000173  |   0.001344     |      758     |     99.25%         |
|Basic Model |     20    |    0.001893	|   0.003710     |      205.06  |     97.68%         |
|Basic Model |     50    |    0.000435	|   0.003278     |      496     |     98.00%         |
