import numpy as np
import torch
import math


class NeuralNetwork:
    """Class Neural Networks for Feed Forward Pass"""

    def build(self, layers_list: list):
        """

        This initializes the dictionary of matrices Theta and dE_dTheta
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
        self.L = len(self.layers_list) - 1  # index of output layer

        # Create dictionary of Theta and dE_dTheta for each set of layers from input to output
        self.Theta = {}  # Theta is a dictionary
        self.dE_dTheta = {}  # dE_dTheta is a dictionary
        self.a = {}     # a is dictionary to store the result of sigmoid output
        self.z = {}     # z is dictionary to store the result of ThetaT*x

        # Create names for dictionary
        self.strs = ["" for x in range(len(self.layers_list) - 1)]  # Create empty array for names in dictionary
        for i in range(0, len(self.layers_list) - 1):  # Create strings for name of Theta and dE_dTheta layers
            self.strs[i] = "Theta(layer" + str(i) + "-layer" + str(i + 1) + ")"

        for index in range(len(self.layers_list) - 1):
            # Increase num of layers by one at each step as BIAS is the extra input node
            self.Theta_np = np.random.normal(0, 1 / math.sqrt(self.layers_list[index]),
                                             (self.layers_list[index+1], self.layers_list[index] + 1))
            self.Theta[self.strs[index]] = torch.from_numpy(self.Theta_np).type(torch.FloatTensor)  # Convert to torch

        self.total_loss = 0.1     # Set default loss value

    def getLayer(self, layer: int):
        """

        :type layer: int (0, 1, ..., n)
        :rtype: 2D FloatTensor
        """
        self.layer = layer
        return self.Theta[self.strs[layer]]  # return the layer pointer from the dictionary

    def forward(self, input: torch.FloatTensor):
        """

        Feed forward pass of Neural Network
        :type input: 1D FloatTensor or 2D FloatTensor
        :rtype 1D FloatTensor or 2D FloatTensor
        """
        self.input = input.t()  # transpose as input is mxn, m=samples and algo works with nxm
        (row, col) = self.input.size()  # Get size of input to decide if it is 1D or 2D Tensor
        # print(row, col)

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

        bias = bias.type(torch.FloatTensor)  # Typecast bias as FloatTensor for concatenation
        self.a[0] = self.input  # Setting the first values for concatenation as the input tensor

        for i in range(len(self.layers_list) - 1):
            self.a[i] = torch.cat((bias, self.a[i]), 0)  # Input concatenated with 1 as the BIAS NODE in dim 0
            theta = self.Theta[self.strs[i]]
            prod = torch.mm(theta, self.a[i])  # Perform matrix multiplication ThetaT*x
            self.z[i + 1] = prod
            self.a[i + 1] = torch.sigmoid(prod)  # Use sigmoid function on matrix product

        return self.a[self.L].t()   # return values from output layer after transpose

    def backward(self, target: torch.FloatTensor, loss: str):
        """

        Back Propagation Pass, calculated de/dtheta
        :type target: 1D FloatTensor or 2D FloatTensor
        :rtype nil
        """
        self.target = target.t()
        self.loss = loss
        'Perform back prop for MSE Cost function'
        if loss == 'MSE':
            # Calculate total error
            self.total_loss = ((self.a[self.L] - self.target).pow(2).sum())/(2*(len(target)))   # average of loss over all samples
            # print(self.total_loss)

            diff_a = self.a[self.L] * (1 - self.a[self.L])  # diff_sigmoid(output layer) = a(L)(1-a(L))
            delta = torch.mul((self.a[self.L] - self.target), diff_a)   # compute delta[L] = (a[L]-y)diff_sigmoid

            for i in range(self.L-1, -1, -1):
                if i == self.L - 1:     # Calculate dE_dTheta directly for Theta(layer(last-1)-layer(last))
                    self.dE_dTheta[i] = torch.mm(self.a[i], delta.t())  # a(l) * delta(l+1); index of a is one more than rest as a[0] is input
                else:   # Calculate dE_dTheta for all layers before final-1
                    index = torch.LongTensor([1, 2])
                    delta = torch.index_select(delta, 0, index)     # Neglect index 0 of delta for bias node
                    self.dE_dTheta[i] = torch.mm(self.a[i], delta.t())   # a(l) * delta(l+1)

                diff_a = self.a[i] * (1-self.a[i])  # diff_sigmoid(output layer) = a(l)(1-a(l))
                x = self.Theta[self.strs[i]].t().mm(delta)
                delta = torch.mul(x, diff_a)              # delta(l)=(Theta(l)_transpose)delta(l+1)*diff_sigmoid
            #print(self.Theta, self.dE_dTheta)

        # Perfom back prop for Cross Entropy - Softmax + Negative Log Likelihood
        elif self.loss == 'CE':
            # This code works with transpose of what was used in MSE
            (row, col) = self.target.t().size()
            self.target = self.target.t()
            #print(self.a[self.L].t())
            #print(self.target)
            #print(self.a[self.L])

            x = self.a[self.L].t()  # x=predicted output
            ex = np.exp(x)      # compute exp(pred_out)
            #print(x,ex)
            esum = ex.sum(1)    # sum over all output nodes
            #print(esum)
            b = np.log(esum)    # compute log (sum(exp(pred_out)))
            #print(b)
            newsum = (self.a[self.L].t()*self.target).sum(1)    # sum((target_probability)(pred_out))
            #print(newsum)
            sample_loss = b - newsum    # calculate loss for each sample
            self.total_loss = sample_loss.sum()/row     # average total loss for all clasess
            #print(sample_loss,self.total_loss)

            delta = self.a[self.L] - self.target   # delta = pred_out - targey
            #print("I am delta %r" %delta)

            for i in range(self.L-1, -1, -1):
                if i == self.L - 1:     # calculate dE_dTheta for Theta(layer(last-1)-layer(last)))
                    #print(self.a[i], self.Theta)
                    self.dE_dTheta[i] = torch.mm(self.a[i], delta.t())  # a(l) * delta(l+1)
                    #print(self.dE_dTheta[i].t())
                else:
                    #index = torch.LongTensor([1, 2])
                    #delta = torch.index_select(delta, 0, index)
                    #print("new iter")
                    diff_a = self.a[i+1] * (1-self.a[i+1])
                    #print(diff_a)
                    #print(self.Theta[self.strs[i]])
                    delta = self.Theta[self.strs[i]].mm(diff_a)
                    #print("end of iter for loop %r"%delta)
                    #print(self.a[i+1], delta.t(), self.a[self.L] - self.target)
                    self.dE_dTheta[i] = torch.mm((self.a[self.L] - self.target).sum()/col, delta.t())  # a(l) * delta(l+1)
                    #print(self.dE_dTheta[i])

    def updateParams(self, eta: float):
        """

        :type eta: float
        :param eta: learning rate
        """
        self.eta = eta
        for index in range(len(self.layers_list) - 1):
            gradient = torch.mul(self.dE_dTheta[index], self.eta)
            # print("grad %r %r"%(gradient, self.Theta[self.strs[index]]))
            self.Theta[self.strs[index]] = self.Theta[self.strs[index]] - gradient.t()









