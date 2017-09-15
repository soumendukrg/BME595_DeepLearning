# **BME 595 DEEP LEARNING**

### **Soumendu Kumar Ghosh**


# **HOMEWORK-03:** *Neural Networks - Back-propagation pass*

The results of different parts of the howework are described in this file.

##### Platform and Packages used:
- **Anaconda** 4.3.25, **Python** 3.6.2, **Pytorch** 0.1.12, **Torchvision** 0.1.8
- Machine: **Intel core i7** (4 cores), **16GB RAM**
- OS: **Ubuntu 14.04** in Virtual Box with 8 GB RAM and 2 cores allocated, 20GB Virtual Disk.

## **INTRODUCTION**
**Back propagation** is a technique used in *Artificial Neural Networks (ANN)* to calculate the gradient of the error (loss function) of a *Neural Network* model with respect to different weights(theta) of the network after a certain batch of input is processed. Stochastic gradient descent is used to reduce the model's error. Specifically, the weights are updated depending upon the gradient of the error w.r.t itself. The network is trained using these techniques until the stopping criteria is reached (which may be order of the total error or a predetermined no of iterations/epochs).

## **OVERVIEW OF THE NETWORK**
In this assignment, we create a full fledged neural network which implements feed forward pass, back propagation pass and gradient descent for weight update. Four different types of logic gates were created using separate instances of the network, they were trained and tested for correctness.

## **API**
### Class: NeuralNetwork
 - The file *neural_network.py* implements this class.
 - The `build(([int] in, [int] h1, [int] h2, …, [int] out))` method initializes the network with input, hidden and output layers. It also initializes random *thetas* from a normal distribution (0 mean, 1/sqrt(layer_size) standard deviation). Additional dictionaries dE_dTheta, a, z have been created.
 - The `getLayer([int] layer)` method returns theta(layer)
 - The `forward([1D/2D FloatTensor] input)` method performs feed forward pass, computes the final output of the network using the inputs and initialized thetas.
 - The `backward([1D/2D FloatTensor] target, [string] loss)` method performs the back propagation pass to calculate dE_dTheta for different layers starting from the last layer and propagating towards the input layer.
   - **BONUS**: Two different cost functions have been used in this method: *Mean Square Error(MSE)* and *Cross Entropy(CE) (Softmax + Negative Log Likelihood)*. The backward method can be called with either one of them from the `logic_gates.py` by passing argument `MSE` or `CE`.
 - The `updateParams([float] eta)` method performs the gradient descent method to update Theta using dE_dTheta computed in the `backward` method.
 
## **Secondary API: logic_gates**
### Classes: AND, OR, NOT, XOR
- The file *logic_gates.py* implements these classes.
- The NeuralNetwork API is used to create 4 different networks implementing the gates **AND, OR, NOT, XOR**.
- Each class implements two public methods, viz. `train` and `forward` apart from the default methods `__init__` and `__call__`
  - For training, several iterarions/epochs are run in the `train()` method until the total loss of the network goes below 0.01 (this is the stopping criterion).
  - In each iteration, the `train` method creates *input data (training data)* on the fly using different permutations of the base dataset `[[0,0], [0,1], [1,0], [1,1]]` and also creates the corresponding target output using Python's and, or, not and combination of these three for AND, OR, NOT, XOR gate respectively. Then it calls these methods from the NeuralNetwork class: `forward(train_data)`, `backward(target, loss)`, and `updateParams(eta)` and then goes to the next iteration.
  - The `forward([boolean]x, [boolean]y)` function of the logic gate is used to find output of the network after it has completed training.
  - `getLayer(layer)` was used inside the `train` method to find trained weights to be used for comparison later on.
  
## Running the tests
A **test.py** file has been additionally provided to show that the network is trained properly and giving proper outputs as expected. All the test cases for 4 gates are implemented in this file. Run the following code (without $ sign) in the terminal.

```
$ python test.py
```
The test file basically performs the following set of codes. Only a single gate has been shown here as an example.
```(python)
from logic_gates import AND
And = AND()  # creating instance of the neural network class
And.train()  # train the network to represent AND gate functionality
And.forward(False, False) # test the trained network
# One can also use And(False, False) which will invoke the __call__ method.
```

## **RESULTS**
Comparison of Theta after network was trained using back propagation and SGD with the handcrafted Theta used in Homework-02
#### **AND Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [-30,  20,  20] | Theta(0) = [-4.9516,  3.2259,  3.2256]|

Sign of the elements and ratio of pair of elements of Theta are almost same in both cases. 

#### **OR Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [-10,  20,  20] | Theta(0) = [-1.3573,  3.3278,  3.3331]|

Sign of the elements and ratio of pair of elements of Theta are almost same in both cases. 

#### **NOT Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [10, -20] | Theta(0) = [1.7216, -3.7065]|

Similar to the last two gates, sign of the elements and ratio of pair of elements of Theta are almost same in both cases. 

#### **XOR Gate**

| Handcrafted Theta | Theta from Trained Network|
|------|------|
|Theta(0) = [[-50,  60, -60], [-50, -60,  60]] | Theta(0) = [[-4.5425,  2.9612,  2.9612], [1.5967, -5.2660, -5.2658]]|
|Theta(1) = [-50, 60,  60] | Theta(1) = [2.9498, -5.8073, -6.0882]|

Here, it is observed that the signs and ratios of the elements of thetas are different. However, the overall gate functionality resembles XOR. Actually the network resembeles an ```(NOR(XY, X`Y`))``` which upon simplification gives XOR. Here is the proof:

**Note**: NOT is represented by **`**, AND by **.** and OR by **+**.

```
Theta(0,0): X AND Y = X.Y
Theta(0,1): X` AND Y`= X`.Y`
Theta(1)  : NOT (X.Y OR X`.Y`) = (X.Y + X`.Y`)` = (X.Y)`.(X`.Y`)` = (X` + Y`).(X + Y) = X.Y` + X`.Y = X XOR Y
```
### Comparison of Error vs Epoch for all gates
The following plot shows the total loss/error of the different gates at each iteration of the training phase with respect to the no. of iterations (epochs). A learning rate of 5.0 was used and we stopped training once the error reduced below 0.01.

![PLOT](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-03/Error_vs_Epoch.png)
 
##### Observations from Plot
- From the plot, we observe that, the NOT gate is trained fastest as it is the most simple network. Next fastest training is OR gate. The XOR gate was next. Last was the AND gate. It should be noted that we could have stopped our training earlier for AND Gate such that it still gave correct output.
- Except XOR, all other gates are linear classifier and only had input and output layer (no hidden layer). So, the network was able to quickly approach low errors, indicated by the steep slope of the curve. However, the XOR gate is a non-linear classifier, hence, the change was more gradual in this case.

#### BONUS:
It was observed that when **Cross Entropy** Loss function was used, less no of iterations were required to complete training. Also, the plots were steeper than in **MSE**, indicating that as loss was high initially, the thetas were penalized more. This is similar to behavior of human brains, as we tend to penalize learning more, if we predict wrongly. In case of MSE, it was opposite, i.e., even if loss was higher, theta still changed slowly. This was evident from the graph of XOR Gate. This was because the derivative of the sigmoid function output was very small, which affected in slow change of theta. In CE, large initial loss led to large change in theta. So, less iterations were required. It is important to note that change of learning rate will again lead to different sort of behavior. This was a good learning experience and in future, other cost functions may be implemented.

