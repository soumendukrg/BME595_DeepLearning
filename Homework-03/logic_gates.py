from neural_network import NeuralNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt


class AND:
    def __init__(self):
        self.and_gate = NeuralNetwork()
        self.and_gate.build([2, 1])  # Create network for AND gate, 2 input layer, 1 output layer
        self.max_iter = 10000   # Set the maximum no of epochs or iterations

    def __call__(self, x, y):
        self.x = x
        self.y = y
        'Test Case after network is trained'
        output = self.forward(self.x, self.y)
        return output


    def train(self):
        print("\nStarting training Network for AND Gate")
        # Create base training dataset
        dataset = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target_data = torch.rand(len(dataset))  # Initialize target dataset randomly
        target_data = torch.unsqueeze(target_data, 1)  # Convert dim from 4 to 4x1 for doing transpose inside NN

        # Train the network repetitively using different permutations of base training dataset
        for i in range(self.max_iter):
            # Create training data for forward function
            index = torch.randperm(4)   # permutation of index among 0, 1, 2, 3
            train_data = torch.index_select(dataset, 0, index)  # permutation of base training dataset

            # Create target data for backward function
            for j in range(len(dataset)):
                target_data[j, :] = train_data[j, 0] and train_data[j, 1] # find target for given order of training data
            # print(i, train_data, target_data)
            # print(self.and_gate.total_loss)

            # Plot epoch vs total_loss
            plt.plot(i, self.and_gate.total_loss, '.r-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)

            #target_data = torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            # Start training
            if self.and_gate.total_loss > 0.01:     # continue training until overall network loss is low enough
                output = self.and_gate.forward(train_data)  # do feed forward pass to find network output
                self.and_gate.backward(target_data, 'MSE')     # do back propagation pass to find dE/dTheta
                self.and_gate.updateParams(1.0)     # update parameters(weights/theta) using learning_rate = 1.0
            else:
                print("Training completed in %d iterations\n" % i)  # i started from 0, so need not do i+1
                break   # break out of for loop as desired loss reached

        #plt.show()
        # Compare new Theta with manually set Theta used in Homework-02
        old_theta = torch.FloatTensor([[-30, 20, 20]])
        new_theta = self.and_gate.getLayer(0)
        print("Manually set Theta: %r\n Newly learned Theta %r\n" % (old_theta, new_theta))

    def forward(self, x: bool, y: bool):
        self.x = x
        self.y = y
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        # No need to convert to int as in Python2 and 3, default value of True is 1 and False is 0
        output = self.and_gate.forward(torch.FloatTensor([[self.x, self.y]]))
        bool_value = (output.numpy())  # Convert the result of the network to numpy
        #print(bool_value)
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test code


class OR:
    def __init__(self):
        self.or_gate = NeuralNetwork()
        self.or_gate.build([2, 1])  # Create network for AND gate, 2 input layer, 1 output layer
        self.max_iter = 10000  # Set the maximum no of epochs or iterations


    def __call__(self, x, y):
        self.x = x
        self.y = y
        'Test Case after network is trained'
        output = self.forward(self.x, self.y)
        return output


    def train(self):
        print("\nStarting training Network for OR Gate")
        # Create base training dataset
        dataset = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target_data = torch.rand(len(dataset))  # Initialize target dataset randomly
        target_data = torch.unsqueeze(target_data, 1)  # Convert dim from 4 to 4x1 for doing transpose inside NN

        # Train the network repetitively using different permutations of base training dataset
        for i in range(self.max_iter):
            # Create training data for forward function
            index = torch.randperm(4)  # permutation of index among 0, 1, 2, 3
            train_data = torch.index_select(dataset, 0, index)  # permutation of base training dataset

            # Create target data for backward function
            for j in range(len(dataset)):
                target_data[j, :] = train_data[j, 0] or train_data[j, 1]  # find target for given order of training data
            # print(i, train_data, target_data)
            # print(self.and_gate.total_loss)

            # Plot epoch vs total_loss
            a = plt.plot(i, self.or_gate.total_loss, '.g-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)


            # Start training
            if self.or_gate.total_loss > 0.01:  # continue training until overall network loss is low enough
                output = self.or_gate.forward(train_data)  # do feed forward pass to find network output
                self.or_gate.backward(target_data, 'MSE')  # do back propagation pass to find dE/dTheta
                self.or_gate.updateParams(5.0)  # update parameters(weights/theta) using learning_rate = 1.0
            else:
                print("Training completed in %d iterations\n" % i)  # i started from 0, so need not do i+1
                break  # break out of for loop as desired loss reached

        # plt.show()
        # Compare new Theta with manually set Theta used in Homework-02
        old_theta = torch.FloatTensor([[-10, 20, 20]])
        new_theta = self.or_gate.getLayer(0)
        print("Manually set Theta: %r\n Newly learned Theta %r\n" % (old_theta, new_theta))

    def forward(self, x: bool, y: bool):
        self.x = x
        self.y = y
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        # No need to convert to int as in Python2 and 3, default value of True is 1 and False is 0
        output = self.or_gate.forward(torch.FloatTensor([[self.x, self.y]]))
        bool_value = (output.numpy())  # Convert the result of the network to numpy
        #print(bool_value)
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test code


class NOT:
    def __init__(self):
        self.not_gate = NeuralNetwork()
        self.not_gate.build([1, 1])  # Create network for AND gate, 2 input layer, 1 output layer
        self.max_iter = 10000  # Set the maximum no of epochs or iterations


    def __call__(self, x):
        self.x = x
        'Test Case after network is trained'
        output = self.forward(self.x)
        return output


    def train(self):
        print("\nStarting training Network for NOT Gate")
        # Create base training dataset
        dataset = torch.FloatTensor([[0], [1]])
        target_data = torch.rand(len(dataset))  # Initialize target dataset randomly
        target_data = torch.unsqueeze(target_data, 1)  # Convert dim from 4 to 4x1 for doing transpose inside NN

        # Train the network repetitively using different permutations of base training dataset
        for i in range(self.max_iter):
            # Create training data for forward function
            index = torch.randperm(2)  # permutation of index among 0, 1, 2, 3
            train_data = torch.index_select(dataset, 0, index)  # permutation of base training dataset

            # Create target data for backward function
            for j in range(len(dataset)):
                target_data[j, :] = not train_data[j, 0]  # find target for given order of training data
            # print(i, train_data, target_data)
            # print(self.and_gate.total_loss)

            # Plot epoch vs total_loss
            plt.plot(i, self.not_gate.total_loss, '.b-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)

            # Start training
            if self.not_gate.total_loss > 0.01:  # continue training until overall network loss is low enough
                output = self.not_gate.forward(train_data)  # do feed forward pass to find network output
                self.not_gate.backward(target_data, 'MSE')  # do back propagation pass to find dE/dTheta
                self.not_gate.updateParams(5.0)  # update parameters(weights/theta) using learning_rate = 1.0
            else:
                print("Training completed in %d iterations\n" % i)  # i started from 0, so need not do i+1
                break  # break out of for loop as desired loss reached

        # plt.show()
        # Compare new Theta with manually set Theta used in Homework-02
        old_theta = torch.FloatTensor([[10, -20]])
        new_theta = self.not_gate.getLayer(0)
        print("Manually set Theta: %r\n Newly learned Theta %r\n" % (old_theta, new_theta))

    def forward(self, x: bool):
        self.x = x
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        # No need to convert to int as in Python2 and 3, default value of True is 1 and False is 0
        output = self.not_gate.forward(torch.FloatTensor([[self.x]]))
        bool_value = (output.numpy())  # Convert the result of the network to numpy
        #print(bool_value)
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test code


class XOR:
    def __init__(self):
        self.xor_gate = NeuralNetwork()
        self.xor_gate.build([2, 2, 1])  # Create network for AND gate, 2 input layer, 1 output layer
        self.max_iter = 100000  # Set the maximum no of epochs or iterations


    def __call__(self, x, y):
        self.x = x
        self.y = y
        'Test Case after network is trained'
        output = self.forward(self.x, self.y)
        return output


    def train(self):
        print("\nStarting training Network for XOR Gate")
        # Create base training dataset
        dataset = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target_data = torch.rand(len(dataset))  # Initialize target dataset randomly
        target_data = torch.unsqueeze(target_data, 1)  # Convert dim from 4 to 4x1 for doing transpose inside NN

        # Train the network repetitively using different permutations of base training dataset
        for i in range(self.max_iter):
            # Create training data for forward function
            index = torch.randperm(4)  # permutation of index among 0, 1, 2, 3
            train_data = torch.index_select(dataset, 0, index)  # permutation of base training dataset

            # Create target data for backward function
            for j in range(len(dataset)):
                target_data[j, :] = ((train_data[j, 0]) and (not train_data[j, 1])) or ((not train_data[j, 0]) and (train_data[j, 1]))
            # print(i, train_data, target_data)
            # print(self.xor_gate.total_loss)

            # Plot epoch vs total_loss
            plt.plot(i, self.xor_gate.total_loss, '.m-')
            plt.xlabel('total_loss')
            plt.ylabel('Epoch')
            plt.title('Total Loss vs Epoch')
            plt.grid(True)

            # Start training
            if self.xor_gate.total_loss > 0.01:  # continue training until overall network loss is low enough
                output = self.xor_gate.forward(train_data)  # do feed forward pass to find network output
                self.xor_gate.backward(target_data, 'MSE')  # do back propagation pass to find dE/dTheta
                self.xor_gate.updateParams(5.0)  # update parameters(weights/theta) using learning_rate = 1.0
            else:
                print("Training completed in %d iterations\n" % i)  # i started from 0, so need not do i+1
                break  # break out of for loop as desired loss reached

        plt.show()
        # Compare new Theta with manually set Theta used in Homework-02
        old_theta1 = torch.FloatTensor([[-50, 60, -60], [-50, -60, 60]])
        old_theta2 = torch.FloatTensor([[-50, 60, 60]])
        new_theta1 = self.xor_gate.getLayer(0)
        new_theta2 = self.xor_gate.getLayer(1)

        print("Manually set Theta: %r %r\n Newly learned Theta %r %r\n" % (old_theta1, old_theta2, new_theta1, new_theta2))

    def forward(self, x: bool, y: bool):
        self.x = x
        self.y = y
        ' TESTING PHASE: Call forward function of NN to do feed forward pass on the trained network '
        # No need to convert to int as in Python2 and 3, default value of True is 1 and False is 0
        output = self.xor_gate.forward(torch.FloatTensor([[self.x, self.y]]))
        bool_value = (output.numpy())  # Convert the result of the network to numpy
        #print(bool_value)
        return bool(np.around(bool_value))  # Convert to Boolean value and return to test code