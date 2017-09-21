# **BME 595 DEEP LEARNING**

### **Soumendu Kumar Ghosh**


# **HOMEWORK-04:** *Neural Networks - Back-propagation pass (MNIST)*

The results of different parts of the howework are described in this file.

##### Platform and Packages used:
- **Anaconda** 4.3.25, **Python** 3.6.2, **Pytorch** 0.1.12, **Torchvision** 0.1.8
- Machine: **Intel core i7** (4 cores), **16GB RAM**
- OS: **Ubuntu 14.04** in Virtual Box with 8 GB RAM and 2 cores allocated, 20GB Virtual Disk.

## **INTRODUCTION**
This assignment deals with training and testing [**The MNIST Database**](http://yann.lecun.com/exdb/mnist/) of handwritten digits using the **NeuralNetwork API** designed in the last assignment as well as Pytorch's nn package. The MNIST database has a training set of 60,000 examples, and a test set of 10,000 examples. In both cases, the network model is trained using the training set and validated using the test set over several no of epochs(iterations).

## **API**
### Class: NeuralNetwork
 - The file *neural_network.py* implements this class. This was implemented in the last assignment.
 - The following methods `build(([int] in, [int] h1, [int] h2, …, [int] out)), getLayer([int] layer), forward([1D/2D FloatTensor] input), backward([1D/2D FloatTensor] target), updateParams([float] eta)` are present in this class.
 
 ### Class: MyImg2Num
  - The file *my_img2num.py* implements this class.
  - `__init__`: The class constructor performs the following tasks:
    - Initializes hyperparameters for training and validation 
    - Downloads the *MNIST* dataset on the host PC (if not already downloaded) and creates the training and testing dataloaders after transforming the images into tensors and enabling shuffling of the data.
    - Initializes the network model with input layer, output layer, and 3 hidden layers (details shown later) and corresponding theta.
  - `train()`: 
    - This method trains the network by performing the feedforward pass, the back propagation pass and updates the thetas(weights) depending on the gradient of thetas. The 2D input tensor is converted to 1D tensor using `view` before feeding to the network model during the forward pass. One-hot encoding is performed on the target labels before feeding to the backward method. The training data is subdivided into several mini batches and training is performed on each batch of data. Subsequently, the average training loss and computation time is recorded for future plotting.
    - After the network is trained for all the batches of data once, the validation dataset is fed to the network. The predicted label(class) of the dataset is compared with the target labels and validation loss and network accuracy is computed.
    - This process of training and validation is repeated for predefined number of epochs.
  - `[int] forward([28x28 ByteTensor] img)`:
    - This method takes a single image from the MNIST dataset in tensor format and performs the forward pass of the trained network. It returns the predicted label(class) of the input image.
 
 ### Class: NnImg2Num
  - The file *nn_img2num.py* implements this class.
  - `__init__`: The class constructor performs the following tasks:
    - Initializes hyperparameters for training and validation
    - Downloads the *MNIST* dataset on the host PC similar to the *MyImg2Num* class.
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
 
## **Results and Observations**
### **MyImg2Num**
#### Loss vs Epoch Comparison
![Plot1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-04/MyImg2Num_LossvsEpoch_30_lr0.1_large_net.png)
It can be observed from the plot that the average training loss and average validation loss is high after the 1st epoch. This also justifies low validation accuracy. They both fall down steeply in the 2nd epoch. This indicates that the network is being properly trainted. The losses reduce slowly and accuracy increases slowly with each epoch. Towards the later epochs, the loss and accuracy is almost constant, indicating saturation.

#### Time vs Epoch Comparison
![Plot2](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-04/MyImg2Num_TimevsEpoch_30_lr0.1_large_net.png)
Though it is expected that time per epoch should be fairly similar, it fluctuates a bit. This may be affected by other running programs in the computer. 

### NnImg2Num
#### Loss vs Epoch Comparison
![Plot3](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-04/NnImg2Num_LossvsEpoch_30_lr15_large_net.png)
Similar to the NeuralNetwork API, initial loss is high, decreases sharply and then gradually, until it saturates. 

#### Time vs Epoch Comparison
![Plot4](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-04/NnImg2Num_TimevsEpoch_30_lr15_large_net.png)
In this case also, time per epoch fluctuates.

### MyImg2Num vs NnImg2Num
#### Average Loss Comparison
It is observed that network built using NeuralNetwork API reaches higher accuracy and lower loss much faster than network built using nn package. Also final accuracy after 30 epochs is slightly better in case of NeuralNetwork API.

|MyImg2Num Learning Rate|Final Accuracy|NnImg2Num Learning Rate|Final Accuracy|
|----|---------|---------|----|
|0.1|98.3|15.0|98.0|

#### Accuracy Comparison
#### Time Comparison
Neural Network API model takes less average computation time per epoch than Pytorch nn package model.

### Observations
#### Learning rate vs Accuracy
It is observed that high learning rate makes the network converge faster, i.e. lower loss and higher accuracy is achieved in less epochs than when using low learning rate. All observations recorded after 30 epochs.

#### Time and Accuracy vs Network size
Lower number of hidden layers reduces training time, however, the final accuracy achieved is slightly less than that achieved with higher no of hidden layers. All observations recorded after 30 epochs

|MyImg2Num Network|Final Accuracy|NnImg2Num Network|Final Accuracy|
|----|---------|---------|----|
|784, 512, 256, 64, 10|98.3|784, 512, 256, 64, 10|98.0|
|784, 512, 64, 10|98.2|784, 512, 64, 10|97.8|




