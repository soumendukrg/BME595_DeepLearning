# **BME 595 DEEP LEARNING**

### **Soumendu Kumar Ghosh**


# **HOMEWORK-03:** *Neural Networks - Back-propagation pass*

The results of different parts of the howework are described in this file.

##### Platform and Packages used:
- **Anaconda** 4.3.25, **Python** 3.6.2, **Pytorch** 0.1.12, **Torchvision** 0.1.8
- Machine: **Intel core i7** (4 cores), **16GB RAM**
- OS: **Ubuntu 14.04** in Virtual Box with 8 GB RAM and 2 cores allocated, 20GB Virtual Disk.

## **INTRODUCTION**
Back propagation is a technique used to calculate the gradient of the error of the Neural Network model with respect to the  different weights(theta) of the network. Then, stochastic gradient descent is used in order to reduce the model's error. The network trained using these two techniques till the stoppin criteria is reached (which may be order or total error).

## **OVERVIEW OF THE NETWORK**
In this assignment, we create a full fledged neural network with feed forward pass, back propagation pass and gradient descent.

## **API**
### Class: NeuralNetwork
 - The file *neural_network.py* implements this class.
 - The `build` method initializes the network with input, hidden and output layers. It also initializes random Theta from a normal distribution (0 mean, 1/sqrt(layer_size) standard deviation).
 - The `forward` method computes the final output using feed forward pass using the inputs and initialized thetas.
 - The `backward` method performs the back propagation pass to calculate dE_dTheta for Theta.
   - Two different cost function have been used: *Mean Square Error* and *Cross Entropy (Softmax + Negative Log Likelihood)*
   - Each of them can be chosen from the logic gates by passing a string MSE or CE.
 - The `updateParams` method performs the stochastic gradient descent method to update Theta using dE_dTheta computed in the `backward` method.
 
 ## **Secondary API: Logic Gates**
### Class: AND, OR, NOT, XOR
- The file *logic_gates.py* implements these classes.
- The NeuralNetwork API is used to create 4 different networks implementing the gates **AND, OR, NOT, XOR**.
- Each class implements two main methods, viz. `train` and `forward`
  - Several iterarions are run in the `train` method until the total loss of the network goes below 0.01 (this is the stopping criterion).
  - In each iteration, the `train` method creates input and target dataset on the fly using different permutations of the dataset `[[0,0], [0,1], [1,0], [1,1]]` and corresponding target output. Then it calls forward, backward, and updateParams and then goes to the next iteration.
  - The forward function is used to find out output of the network after it is trained.
  
## Running the tests
A test.py file has been additionally provided to show that the network is trained properly and giving proper outputs as excepted. All the test cases for 4 gates are implemented in *test.py*. Run the following code (without $ sign) in the terminal.

```
$ python test.py
```
The test file basically performs the following set of codes. Only a single gate has been shown here as an example.
```
from logic_gates import AND
And = AND()
And.train()
And.forward(False, False)
# One can also use And(False, False) which will invoke the __call__ method.
```


## **RESULTS**
Comparison of Theta after network was trained using back propagation and SGD with the handcrafted Theta used in Homework-02
#### **AND Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [-30,  20,  20] | Theta(0) = [-4.9516,  3.2259,  3.2256]|

The signs of theta and ratio of values are similar.

#### **OR Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [-10,  20,  20] | Theta(0) = [-1.3573,  3.3278,  3.3331]|

The signs of theta and ratio of values are similar.

#### **NOT Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [10, -20] | Theta(0) = [1.7216, -3.7065]|

The signs of theta and ratio of values are similar.

#### **XOR Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [[-50,  60, -60], [-50, -60,  60]] | Theta(0) = [[-4.5425,  2.9612,  2.9612], [1.5967, -5.2660, -5.2658]]|
|Theta(1) = [-50, 60,  60] | Theta(1) = [2.9498, -5.8073, -6.0882]|

Here, it is observed that the signs of thetas are changed but the overall gate is the same. Actually the network resembeles an `(NOR(XY, X`Y`))` which gives XOR.

The following plot shows the total loss of the different gates with respect to the no of iterations (epochs)
The color of plots represents the following gates:
- Red: AND
- Green: OR
- Blue: NOT
- Magenta: XOR

![PLOT](https://ml4a.github.io/images/figures/sigmoid.png)

#### BONUS:
It was observed that using Cross Entropy Loss function resulted in less no of iterations to completed training. Also, the plots were steeper, that indicates if loss was high, then thetas will be penalizd more.

