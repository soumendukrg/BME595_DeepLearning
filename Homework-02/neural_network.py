import numpy as np
import torch
import math


class NeuralNetwork:
    """Class Neural Networks for Feed Forward Pass"""

    def __init__(self, layers_list: list):
        """

        This initializes the dictionary of matrices theta
        :type layers_list: list
        :rtype: nil
        """
        self.layers_list = layers_list
        if len(self.layers_list) < 2:  # Error Checking
            print("\nERROR: The network must have at least 2 layers!\n")
            exit(1)
        # Extract number of nodes in input and output layers from input list
        # Not required, kept for future use
        self.numNodes_input = self.layers_list[0]
        self.numNodes_output = self.layers_list[len(self.layers_list) - 1]

        # Create dictionary of theta for each set of layers from input to output
        self.theta = {}  # theta is a dictionary
        # Create names for dictionary
        self.strs = ["" for x in range(len(self.layers_list) - 1)]  # Create empty array for names in dictionary
        for i in range(0, len(self.layers_list) - 1):  # Create strings for name of theta layers
            self.strs[i] = "theta(layer" + str(i) + "-layer" + str(i + 1) + ")"

        for index in range(len(self.layers_list) - 1):
            # Increase num of layers by one at each step as BIAS is the extra input node
            self.theta_np = np.random.normal(0, 1 / math.sqrt(self.layers_list[index]),
                                             (self.layers_list[index] + 1, self.layers_list[index + 1]))
            self.theta[self.strs[index]] = torch.from_numpy(self.theta_np)  # Convert to torch
            # print(self.theta)

    def getLayer(self, layer: int):
        """

        :type layer: int (0, 1, ..., n)
        :rtype: 2D DoubleTensor
        """
        self.layer = layer
        # print(self.strs[layer], self.layer)
        return self.theta[self.strs[layer]]  # return the layer pointer from the dictionary

    def forward(self, input: torch.DoubleTensor):
        """

        Feed forward pass of Neural Network
        :type input: 1D DoubleTensor or 2D DoubleTensor
        :rtype 1D DoubleTensor or 2D DoubleTensor
        """

        # Define sigmoid function
        def sigmoid(inp: torch.DoubleTensor):
            product = inp.numpy()  # Convert from torch to numpy
            sig = 1 / (1 + np.exp(-product))
            return torch.from_numpy(sig)  # Convert to torch

        self.input = input
        (row, col) = self.input.size()  # Get size of input to decide if it is 1D or 2D Tensor

        if row != self.numNodes_input:  # Error checking
            print("ERROR: The defined network input layer and input size mismatch!")
            print("Please enter only %r no of inputs" % self.numNodes_input)
            exit(2)

        # Create bias nodes in 1D or 2D
        if col == 1:
            bias = torch.ones((1, 1))
        else:
            bias = torch.ones((1, col))  # Each row represents n batches of 1 input, hence add 1 row of ones as bias,
            # as if n batches of ones

        bias = bias.type(torch.DoubleTensor)  # Typecast bias as DoubleTensor for concatenation
        sig_prod = self.input  # Setting the first values for concatenation as the input tensor

        for i in range(len(self.layers_list) - 1):
            cat_input = torch.cat((bias, sig_prod), 0)  # Input concatenated with 1 as the BIAS NODE in dim 0
            theta_trans = torch.t(self.theta[self.strs[i]])  # Transpose theta for multiplication
            prod = torch.mm(theta_trans, cat_input)  # Perform matrix multiplication thetaT*x
            sig_prod = sigmoid(prod)  # Use sigmoid function on matrix product

        return sig_prod
