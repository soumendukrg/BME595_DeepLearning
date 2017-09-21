import numpy as np
import torch
import math


class NeuralNetwork:
    """Class Neural Networks for Feed Forward Pass"""

    def __init__(self):
        pass

    def build(self, layers_list: list):
        """

        This initializes the dictionary of matrices Theta and dE_dTheta
        :type layers_list: list
        :rtype: nil
        """
        self.layers_list = layers_list
        self.numLayers = len(self.layers_list)
        if self.numLayers < 2:  # Error Checking
            print("\nERROR: The network must have at least 2 layers!\n")
            exit(1)

        self.L = self.numLayers - 1  # index of last (output) layer

        # Create dictionary of Theta and dE_dTheta for each set of layers from input to output
        self.Theta = {}  # Theta is a dictionary, index from 0 to numLayers-2, nL=4, index=0,1,2
        self.dE_dTheta = {}  # dE_dTheta is a dictionary, index from 0 to numLayers-2, index=0,1,2
        self.a = {}  # a is dictionary to store the result of sigmoid output, index from 0 to nL-1
        self.z = {}  # z is dictionary to store the result of Theta*x, index from 1 to nL-1

        # Initialize Theta
        for i in range(0, self.numLayers - 1):  # range goes from 0 to self.numLayers-2
            # Increase num of layers by one at each step as BIAS is the extra input node
            # theta_row = len(a(nextLayer)) theta_col = len(a(currentLayer))+1 as num of col=len(a(0))+bias
            theta_np = np.random.normal(0, 1 / math.sqrt(self.layers_list[i]),
                                        (self.layers_list[i + 1], self.layers_list[i] + 1))
            self.Theta[i] = torch.from_numpy(theta_np).type(torch.FloatTensor)  # Convert to torch

        self.total_loss = 0.0  # Set default loss value

    def getLayer(self, layer: int):
        """

        :type layer: int (0, 1, ..., n)
        :rtype: 2D FloatTensor
        """
        return self.Theta[layer]  # return the layer pointer from the dictionary

    def forward(self, input: torch.FloatTensor):
        """

        Feed forward pass of Neural Network
        :type input: 1D FloatTensor or 2D FloatTensor
        :rtype 1D FloatTensor or 2D FloatTensor
        """
        self.input = input.t()  # transpose as input is mxn, m=no of samples and algorithm works with nxm
        (row, col) = self.input.size()  # Get size of input to decide if it is 1D or 2D Tensor, row = n, col = m

        if row != self.layers_list[0]:  # Error checking
            print("ERROR: The defined network input layer and input size mismatch!")
            print("Please enter only %r no of inputs" % self.layers_list[0])
            exit(2)

        # Create bias nodes in 1D or 2D
        if col == 1:
            bias = torch.ones((1, 1))
        else:
            bias = torch.ones((1, col))  # Each row represents n batches of 1 input, hence add 1 row of ones as bias,
            # as if n batches of ones

        bias = bias.type(torch.FloatTensor)  # Typecast bias as FloatTensor for concatenation
        self.a[0] = self.input  # Setting the first values for concatenation as the input tensor

        for l in range(0, self.numLayers - 1):  # i = 0,1,2,...,nL-2
            self.a[l] = torch.cat((bias, self.a[l]), 0)  # Input concatenated with 1 as the BIAS NODE in dim 0
            self.z[l + 1] = torch.mm(self.Theta[l], self.a[l])  # Perform matrix multiplication Theta*x
            self.a[l + 1] = torch.sigmoid(self.z[l + 1])  # Use sigmoid function on matrix product

        return self.a[self.L].t()  # return values from output layer after transpose

    def backward(self, target: torch.FloatTensor):
        """

        Back Propagation Pass, calculated de/dTheta
        :type target: 1D FloatTensor or 2D FloatTensor
        :rtype nil
        """
        self.target = target.t()
        (row, col) = self.target.size()

        'Perform back prop for MSE Cost function'
        # Calculate total error
        self.total_loss = ((self.a[self.L] - self.target).pow(2).sum()) / (2 * col)  # average of loss over all samples
        # diff_sigmoid(output_layer) = a(L)(1-a(L)), self.L = self.numLayers - 1
        diff_a = self.a[self.L] * (1 - self.a[self.L])
        delta = torch.mul((self.a[self.L] - self.target), diff_a)  # compute delta[L] = (a[L]-y).diff_sigmoid

        for l in range(self.numLayers - 2, -1, -1):
            if l == self.numLayers - 2:  # Calculate dE_dTheta(L-1) directly
                self.dE_dTheta[l] = torch.mm(delta, self.a[
                    l].t())  # delta(l+1)a(l)_transpose; index of a is one more than rest as a[0] is input

            else:  # Calculate dE_dTheta for all layers before final-1
                delta = delta.narrow(0, 1, delta.size(0) - 1)  # Neglect index 0 of delta for bias node
                self.dE_dTheta[l] = torch.mm(delta, self.a[l].t())  # delta(l+1)a(l)_transpose

            diff_a = self.a[l] * (1 - self.a[l])  # diff_sigmoid(output layer) = a(l)(1-a(l))
            delta = torch.mul(self.Theta[l].t().mm(delta),
                              diff_a)  # delta(l)=(Theta(l)_transpose)delta(l+1).diff_sigmoid
            # print(self.Theta, self.dE_dTheta)

    def updateParams(self, eta: float):
        """

        :type eta: float
        :param eta: learning rate
        """
        for index in range(0, self.numLayers - 1):
            gradient = torch.mul(self.dE_dTheta[index], eta)
            self.Theta[index] = self.Theta[index] - gradient
