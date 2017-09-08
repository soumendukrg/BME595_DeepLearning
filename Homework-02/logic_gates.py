from neural_network import NeuralNetwork
import torch
import numpy as np


class AND:
    def __init__(self):
        self.and_gate = NeuralNetwork([2, 1])  # Create network for AND gate, 2 ip layer, 1 op layer
        self.theta = self.and_gate.getLayer(0)  # Get randomly initialized theta from the network
        # Set weights of neural network for AND gate
        self.theta.fill_(0)  # Set all values to zero
        self.theta += torch.DoubleTensor([[-30], [20], [20]])  # Update weights for theta inside the network

    def __call__(self, x: bool, y: bool):
        self.x = x
        self.y = y
        output = self.forward()  # Call forward function of and_gate
        bool_value = (output.numpy())  # Convert the result of the network to numpy
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test code

    def forward(self):
        """Call forward function of NeuralNetwork for do forward propagation pass on the network built using the
        parameters in __init__ """
        # No need to convert to int as in Python2 and 3, default value of True is 1 and False is 0
        result = self.and_gate.forward(torch.DoubleTensor([[self.x], [self.y]]))
        return result


class OR:
    def __init__(self):
        self.or_gate = NeuralNetwork([2, 1])
        self.theta = self.or_gate.getLayer(0)  # Get randomly initialized theta from the network
        # Set weights of neural network for AND gate
        self.theta.fill_(0)  # Set all values to zero
        self.theta += torch.DoubleTensor([[-10], [20], [20]])  # Update weights for theta inside the network

    def __call__(self, x: bool, y: bool):
        self.x = x
        self.y = y
        output = self.forward()
        bool_value = (output.numpy())  # Convert to numpy
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test.py

    def forward(self):
        """Call forward function of NeuralNetwork for do forward propagation pass on the network built using the
        parameters in __init__ """
        result = self.or_gate.forward(torch.DoubleTensor([[self.x], [self.y]]))
        return result


class NOT:
    def __init__(self):
        self.not_gate = NeuralNetwork([1, 1])  # Initialize network with 2 input and 1 output node
        self.theta = self.not_gate.getLayer(0)  # Get randomly initialized theta from the network
        # Set weights of neural network for AND gate
        self.theta.fill_(0)  # Set all values to zero
        self.theta += torch.DoubleTensor([[10], [-20]])  # Update weights for theta inside the network

    def __call__(self, x: bool):
        self.x = x
        output = self.forward()
        bool_value = (output.numpy())  # Convert to numpy
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test.py

    def forward(self):
        """Call forward function of NeuralNetwork for do forward propagation pass on the network built using the
        parameters in __init__ """
        result = self.not_gate.forward(torch.DoubleTensor([[self.x]]))
        return result


class XOR:
    def __init__(self):
        self.xor_gate = NeuralNetwork([2, 2, 1])  # input layer=2, hidden layer=2, output layer=1
        self.theta1 = self.xor_gate.getLayer(0)  # Get randomly initialized theta from the network
        self.theta2 = self.xor_gate.getLayer(1)
        # Set weights of neural network for XOR gate
        self.theta1.fill_(0)  # Set all values to zero
        self.theta1 += torch.DoubleTensor([[-50, -50], [60, -60], [-60, 60]])  # Update weights for XY` AND X`Y
        self.theta2.fill_(0)  # Set all values to zero
        self.theta2 += torch.DoubleTensor([[-50], [60], [60]])  # Update weights for XY` + X`Y

    def __call__(self, x: bool, y: bool):
        self.x = x
        self.y = y
        output = self.forward()
        bool_value = (output.numpy())  # Convert to numpy
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test.py

    def forward(self):
        """Call forward function of NeuralNetwork for do forward propagation pass on the network built using the
        parameters in __init__ """
        result = self.xor_gate.forward(torch.DoubleTensor([[self.x], [self.y]]))
        return result
