# **BME 595 DEEP LEARNING**

### **Soumendu Kumar Ghosh**


# **HOMEWORK-02:** *Neural Networks - Feedforward pass*

The results of different parts of the howework are described in this file.

##### Platform and Packages used:
- **Anaconda** 4.3.25, **Python** 3.6.2, **Pytorch** 0.1.12, **Torchvision** 0.1.8
- Machine: **Intel core i7** (4 cores), **16GB RAM**
- OS: **Ubuntu 14.04** in Virtual Box with 8 GB RAM and 2 cores allocated, 20GB Virtual Disk.

## **INTRODUCTION**
**Artificial Neural Network** (ANN) is a computer system model which draws inspiration from the vast network of neurons in animal brain. The goal of an ANN is to learn from the experience of its own interaction with the environment and finally develop perception to predict observations about the environment using its own learned knowledge.
  A feedforward neural network is a simple version of the ANN. This network is made of a single layer of input nodes and a single layer of output nodes, which are intersped by a number of hidded layers. In this network, the information moves only from the input layer to the output layer, hence the name feedforward.
  
## **OVERVIEW OF THE NETWORK**
In this assignment, we create a feedforward neural network with different layers. In such a network, different weights (Θ) are assigned to the connections between each set of layers. Once, the input is fed to the input layer, it is multiplied with pre-assigned weight.

`z = (transpose of Θ)*input`

A logistic function (*sigmoid(z)*) is applied to the product and the output of this function is fed to the different nodes of the next layer (which may be the output layer or a hidden layer). The sigmoid function introduces non-linearities.

![Sigmoid](https://ml4a.github.io/images/figures/sigmoid.png)

This process is followed for all the subsequent layers, until final result is updated in the output node(s). It must be noted that a **Bias** node is added in each layer from the input layer till the layer before the output layer. This is the general framework of the network implemented here. Two APIs have been designed as described below.

## **API**
### Class: NeuralNetwork
- The file *neural_network.py* implements this class.
- The NeuralNetwork class builds the network given a set of layers. The class when initialized with a list (in, h1, h2, ..., out) *creates a dictionary of matrices, Θ*. The network dictionary is populated with Θ(layer) matrices (mapping layer to layer + 1), and each Θ is initialised to random values (mean=0.0, standard deviation=1/sqrt(layer_size)).
- The method `getLayer(layer)` returns Θ(layer), which is similar to a pointer in C. Hence, when this Θ is modified in the second API, the Θ inside the neural network instance is modified as well. This is an example of *call by reference*.
- The method `forward(1D/2D DoubleTensor input)` does the actual feedforward pass of the network as described in the *overview* section. If the input type is a 2D DoubleTensor, it should adhere to the format `D_in x N` where *D_in* is the number of nodes in the input layer and *N* is the batch size. As mentioned earlier, *Bias* term is added in each layer, the size of which depends upon the number of columns of the input, i.e. the batch size *N*. So if only one batch is present, then a single node of Bias is added to the existing nodes in the layer.

## **Secondary API: Logic Gates**
### Class: AND, OR, NOT, XOR
- The file *logic_gates.py* implements these classes.
- The NeuralNetwork API is used to create 4 different networks implementing the gates **AND, OR, NOT, XOR**. These gates take in *boolean* input and gives *boolean* output. In Python 2 and 3, boolean input is stored in *integer* format by default.
- Each class constructor creates a network and sets the weights Θ of the layers of the corresponding network. The weights were calculated by hand for correct logic operation.
- The `__call__` method calls the `forward` method of the class, which in turn calls the `forward` method of NeuralNetwork. The output is converted to boolean and printed.

## Running the tests
All the test cases for 4 gates are implemented in *test.py*. Run the following code (without $ sign) in the terminal.

```
$ python test.py
```

## **RESULTS**
#### **AND Gate**

| Input|Output|
|------|------|
|And(False, False) | False|
|And(False, True) | False|
|And(True, False) | False|
|And(True, True) | True|

#### **OR Gate**

| Input|Output|
|------|------|
|Or(False, False) | False|
|Or(False, True) | True|
|Or(True, False) | True|
|Or(True, True) | True|

#### **NOT Gate**

| Input|Output|
|------|------|
|Not(False) | True|
|Not(True) | False|

#### **XOR Gate**

| Input|Output|
|------|------|
|Xor(False, False) | False|
|Xor(False, True) | True|
|Xor(True, False) | True|
|Xor(True, True) | False|

Tests using 2D tensor input for the gates had also been carried out, but not shown here. In that case, the input has to be in form of a list of dimensions 2xN, where N is the no of batches. In case of boolean gates, since only 4 combinations are possible, n can be maximum 4.
