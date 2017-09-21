import torch
from torchvision import datasets, transforms
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import time


class MyImg2Num:
    def __init__(self):
        """

        Initialize neural network, hyper-parameters, and MNIST data loaders
        """
        # Hyper-parameters for Training
        self.train_batch_size = 60  # total 60000 images, 1000 batches of 60 images each
        self.validation_batch_size = 1000  # total 10000 images, 10 batches of 1000 images each
        self.learning_rate = 0.1  # learning rate for mini-batch gradient descent
        self.epochs = 30  # no of times training and validation to be performed on network

        # Image size parameters
        row = 28  # number of rows of input image
        col = 28  # number of columns of input image
        self.size1D = row * col  # The 2D image has to be converted to 1D tensor, each row has to be placed along the
        #  column dimension one after another, size1D is the no of columns of the converted 1D tensor
        self.labels = 10  # no of output labels (classes), this is no of output layers as well

        'Download MNIST dataset (if not already downloaded), transform from PIL Image to Tensor, enable shuffling'
        # Training dataset loader: when called, it will create shuffled train data of train_batch_size
        # ../data represents root directory where raw data is downloaded to and processed data is loaded from
        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.train_batch_size, shuffle=True)

        # Testing dataset: when called, it will create shuffled test data of test_batch_size
        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.validation_batch_size, shuffle=True)

        # Create neural network model using own created NeuralNetwork API
        # Initialize network with different layers and theta
        # Input layer=784, hidden layers are chosen power of 2, output layer = no of output classes(labels), here 10
        self.nn_model = NeuralNetwork()  # create instance of neural network
        self.nn_model.build([self.size1D, 512, 256, 64, self.labels])

    def train(self):
        """
        This method performs training the network using MNIST dataset. There're 2 main methods: training and validation.
        These two methods are called for a predefined no of epochs.
        training: The model is trained by calling forward, backward, and updateParams in a cycle for all the batches of
            training data. The data is converted into 1D tensor before feeding into forward pass. Onehot encoding is
            performed on the labels for each batch of data and then fed into backward pass. Finally, gradient descent
            is performed to update theta. Loss is found at the end.
        validation: Once the network is trained, validation data obtained from the testing data loader is fed to the
            network and the output is compared with the target labels from the dataset to find out loss and accuracy.
        :return: nil
        """

        def training(epoch):  # training method
            def onehot_training():  # onehot encoder for training labels
                labels_onehot = torch.zeros(self.train_batch_size,
                                            self.labels)  # initialize labels_onehot with all zeros
                for i in range(self.train_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            training_loss = 0  # initialize total training loss for each epoch

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                # Feed forward pass: convert data from 60x1x28x28 to 60x784 and and then compute predicted output
                # by passing current batch of data to the model
                output = self.nn_model.forward(data.view(self.train_batch_size, self.size1D))

                # Backward pass: convert target labels to onehot and compute gradient of the loss with respect to theta
                self.nn_model.backward(onehot_training())
                training_loss += self.nn_model.total_loss  # add current batch loss to total loss

                # Update theta(weights)
                self.nn_model.updateParams(self.learning_rate)
                """ 
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.
                      format(epoch, (batch_id + 1) * self.train_batch_size, len(self.train_data_loader.dataset),
                             100.0 * (batch_id + 1) * self.train_batch_size / len(self.train_data_loader.dataset),
                             self.nn_model.total_loss))"""

            # average loss = sum of loss over all batches/num of batches
            average_training_loss = training_loss / (len(self.train_data_loader.dataset) / self.train_batch_size)
            print("\nTrain Epoch {}: Average loss: {:.6f}".format(epoch, average_training_loss))
            return average_training_loss

        def validation(epoch):  # validation method
            def onehot_validation(target):  # onehot encoder for validation labels
                labels_onehot = torch.zeros(self.validation_batch_size,
                                            self.labels)  # initialize labels_onehot with all zeros
                for i in range(self.validation_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            validation_loss = 0  # initialize loss for whole validation dataset
            total_correct = 0  # no of correct classifications

            for data, target in self.validation_data_loader:
                # Feed forward pass: convert data from 1000x1x28x28 to 1000x784 and then compute predicted output
                # by passing current batch of data to the model
                output = self.nn_model.forward(data.view(self.validation_batch_size, self.size1D))

                # Add MSE loss for each batch of validation data
                validation_loss += ((onehot_validation(target) - output).pow(2).sum()) * 0.5

                value, index = torch.max(output, 1)  # get index of max value among output class
                for i in range(0, self.validation_batch_size):
                    if index[i][0] == target[i]:  # if index equal to target label, correct classification
                        total_correct += 1

            # average loss = sum of loss over all data/num of data image(=10000)
            average_validation_loss = validation_loss / len(self.validation_data_loader.dataset)

            accuracy = 100.0 * total_correct / (len(self.validation_data_loader.dataset))  # calculate total accuracy

            print('\nValidation Epoch {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.1f}%)\n'.
                  format(epoch, average_validation_loss, total_correct, len(self.validation_data_loader.dataset),
                         accuracy))
            print('-----------------------------------------------------------------------------------')

            return average_validation_loss

        'Actual Code starts here, code above are local methods'

        print("\nStarting training of neural network using MyImg2Num on MNIST dataset\n")
        # Perform training and validation in each iteration(epoch)
        epoch_num = range(1, self.epochs + 1)
        train_loss = list()
        validation_loss = list()
        computation_time = list()

        for i in range(1, self.epochs + 1):
            # Training
            start_time = time.time()
            train_loss.append(training(i))
            end_time = time.time() - start_time
            computation_time.append(end_time)
            print('\nTrain Epoch {}: Computation Time: {:.2f} seconds'.format(i, end_time))

            # Validation
            validation_loss.append(validation(i))

        # Plot loss vs epoch
        plt.figure(1)
        plt.plot(epoch_num, train_loss, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Loss')
        plt.plot(epoch_num, validation_loss, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Loss')
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        title = 'Loss vs Epochs using NeuralNetwork API (learning rate %r,train batch size %r,validation batch size %r)'\
                % (self.learning_rate, self.train_batch_size, self.validation_batch_size)
        plt.title(title, fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()

        # Plot time vs epoch
        plt.figure(2)
        plt.plot(epoch_num, computation_time, color='red', linestyle='solid', linewidth='2.0',
                 marker='o', markerfacecolor='red', markersize='5', label='Training Time per epoch')
        plt.ylabel('Computation Time (in seconds)', fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        title = 'Computation Time vs Epochs using NeuralNetwork API (learning rate %r,train batch size %r,validation ' \
                'batch size %r)' % (self.learning_rate, self.train_batch_size, self.validation_batch_size)
        plt.title(title, fontsize=18)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.show()

        # Test forward method
        # label = self.forward(self.train_data_loader.dataset[0][0])
        # print(label, self.train_data_loader.dataset[0][1])

    def forward(self, img: torch.ByteTensor):
        """

        This method takes an image from the MNIST dataset and predict the label of the image using the trained neural
        network model built using NeuralNetwork API
        :param img: 28x28 ByteTensor
        :return: [int] predicted label(class)
        """
        output = self.nn_model.forward(img.view(1, self.size1D))  # Forward pass using trained model
        value, pred_label = torch.max(output, 1)  # get index of max value among output class
        return pred_label
