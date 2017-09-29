import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil, os


class NnImg2Num:
    def __init__(self):
        """

        Initialize neural network, hyper-parameters, and MNIST data loaders
        """
        # Hyper-parameters for Training
        self.train_batch_size = 60  # total 60000 images, 1000 batches of 60 images each
        self.validation_batch_size = 1000  # total 10000 images, 10 batches of 1000 images each
        self.learning_rate = 15.0  # learning rate for mini-batch gradient descent
        self.epochs = 50  # no of times training and validation to be performed on network

        # Image size parameters
        row = 28  # number of rows of input image
        col = 28  # number of columns of input image
        # The 2D image has to be converted to 1D tensor, each row has to be placed along the column dimension one after
        # another, size1D is the no of columns of the converted 1D tensor
        self.size1D = row * col
        self.labels = 10  # no of output labels (classes), this is no of output layers as well

        'Download MNIST dataset (if not already downloaded), transform from PIL Image to Tensor, enable shuffling'
        # Training dataset loader: when called, it will create shuffled train data of train_batch_size
        # ../MNIST_data represents root directory where raw data is downloaded to and processed data is loaded from
        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../MNIST_data', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.train_batch_size, shuffle=True)

        # Validation dataset: when called, it will create shuffled validation data of validation_batch_size
        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../MNIST_data', train=False, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=self.validation_batch_size, shuffle=True)

        torch.manual_seed(1)  # Seed generation for random parameter (theta) generation, guarantees same theta always
        # Create neural network model and loss function using nn package
        # Input layer=784, hidden layers are chosen power of 2, output layer = no of output classes(labels), here 10
        self.nn_model = nn.Sequential(
            nn.Linear(self.size1D, 512), nn.Sigmoid(),
            nn.Linear(512, 256), nn.Sigmoid(),
            nn.Linear(256, 64), nn.Sigmoid(),
            nn.Linear(64, self.labels), nn.Sigmoid()
        )

        # Set loss function as Mean Square Error
        self.loss_fn = nn.MSELoss()  # By default, the losses are averaged over observations for each mini batch

        # We define an Optimizer using the optim package. This will update the theta (weights) of the model.
        # We use SGD as the optimization algorithm. The first argument to the SGD constructor tells the
        # optimizer which Variables it should update, lr specifies the learning rate.
        self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=self.learning_rate)

        # Initialize list of epochs for plotting
        self.epoch_num = range(1, self.epochs + 1)

        # Load model from the latest saved checkpoint and Resume training
        load_chkpt_file = 'NnImg2Num_model_checkpoint.pth.tar'  # Filename to load checkpoint from
        if os.path.isfile(load_chkpt_file):  # check if provided file exists
            print('\nLoading from checkpoint file: {}\n'.format(load_chkpt_file))
            chkpt = torch.load(load_chkpt_file)
            # load state of last training
            self.start_epoch = chkpt['epoch']
            self.best_accuracy = chkpt['best_accuracy']

            # load model parameters
            self.nn_model.load_state_dict(chkpt['state_dict'])
            self.optimizer.load_state_dict(chkpt['optimizer'])

            # load result parameters
            self.train_loss = chkpt['train_loss']
            self.validation_loss = chkpt['validation_loss']
            self.accuracy = chkpt['accuracy']
            self.computation_time = chkpt['time']
            print('Completed loading from checkpoint {}, (last saved epoch {}, best accuracy till now {:.2f})'.
                  format(load_chkpt_file, self.start_epoch, self.best_accuracy))
        else:
            print('\nNo checkpoint to load from\n')
            # Initialize epoch number and starting accuracy (which is 0, when network is untrained)
            self.start_epoch = 0
            self.best_accuracy = 0
            # Initialize results' lists for plotting
            self.train_loss = list()
            self.validation_loss = list()
            self.accuracy = list()
            self.computation_time = list()

    def train(self):
        """
        This method trains the network model using torch nn package using MNIST dataset.
        There're two main methods: training and validation.
        These two methods are called for a predefined no of epochs.
        training: The model is trained by doing forward pass by passing data to model, finding average batch loss,
            doing backward pass to compute gradients, and updating model parameters (theta) in cycle for all batches of
            training data. The data is converted into 1D tensor before feeding into forward pass. One-hot encoding is
            performed on the labels for each batch of data. Final average training loss is found at end of epoch.
        validation: Once the network is trained, validation data obtained from the validation data loader is fed to the
            network and the output is compared with the target labels from the dataset to find out loss and accuracy.
        :return: nil
        """
        def save_checkpoint(state, better, file='NnImg2Num_model_checkpoint.pth.tar'):
            """
            Local method: This method will save a checkpoint of model parameters and results at end of each epoch
            :param state: dictionary of parameters to save
            :param better: True if current epoch accuracy is better than last saved best accuracy
            :param file: name of file where checkpoint is to be saved
            """
            torch.save(state, file)  # save model state in checkpoint file
            if better:  # if current state is better than all other previous state, save in best model file
                shutil.copyfile(file, 'NnImg2Num_best_model.pth.tar')

        def training(epoch):
            """
             Local method: Training of model
             :param epoch: number indicating current epoch
             :return: average training loss for current epoch
             """
            def onehot_training():
                """
                Local method: This method performs onehot encoding for training labels
                :return: encoded labels for the whole batch of data
                """
                labels_onehot = torch.zeros(self.train_batch_size, self.labels)  # initialize labels with all zero
                for i in range(self.train_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            self.nn_model.train()  # Sets the module in training mode
            training_loss = 0  # initialize total training loss for each epoch

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                target = onehot_training()  # convert target labels to onehot labels
                # Wrap data and target in Variable, no gradient required for target
                data, target = Variable(data), Variable(target, requires_grad=False)

                # Feed forward pass: convert data from 60x1x28x28 to 60x784 and then compute predicted output
                # by passing current batch of data to the model
                output = self.nn_model(data.view(self.train_batch_size, self.size1D))

                batch_loss = self.loss_fn(output, target)  # compute average MSE loss for current batch
                training_loss += batch_loss.data[0]  # add current batch loss to total loss

                # We use the optimizer to zero all the gradients for the variables (theta/weights of the model)
                # before the backward pass.
                self.optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters (theta)
                batch_loss.backward()

                # Update the model parameters (theta)
                self.optimizer.step()

                print('\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.
                      format(epoch, (batch_id + 1) * self.train_batch_size, len(self.train_data_loader.dataset),
                             100.0 * (batch_id + 1) * self.train_batch_size / len(self.train_data_loader.dataset),
                             batch_loss.data[0]), end='')

            # average loss = sum of loss over all batches/num of batches
            average_training_loss = training_loss / (len(self.train_data_loader.dataset) / self.train_batch_size)
            print("\t Average Training loss: {:.6f}".format(average_training_loss), end="")
            return average_training_loss

        def validation(epoch):
            """
            Local method: Validation of model
            :param epoch: number indicating current epoch
            :return: average validation loss for current epoch
            """
            def onehot_validation(target):
                """
                Local method: This method performs onehot encoding for validation labels
                :param: input is a batch of labels from validation dataset of size 1000
                :return: encoded labels for the whole batch of data
                """
                labels_onehot = torch.zeros(self.validation_batch_size, self.labels)  # initialize labels with all zeros
                for i in range(self.validation_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            self.nn_model.eval()  # Sets the module in evaluation mode
            validation_loss = 0  # initialize total validation loss for whole validation dataset
            total_correct = 0  # no of correct classifications

            for data, target in self.validation_data_loader:
                onehot_target = onehot_validation(target)  # convert target labels to onehot labels
                # wrap data and onehot_target in Variable, no gradient required for onehot_target
                data, onehot_target = Variable(data), Variable(onehot_target, requires_grad=False)

                # Feed forward pass: convert data from 1000x1x28x28 to 1000x784 and then compute predicted output
                # by passing current batch of data to the model
                output = self.nn_model(data.view(self.validation_batch_size, self.size1D))

                batch_loss = self.loss_fn(output, onehot_target)  # compute average MSE loss for current batch
                validation_loss += batch_loss.data[0]  # add current batch loss to total loss

                value, index = torch.max(output.data, 1)  # get index of max value among output class
                for i in range(0, self.validation_batch_size):
                    if index[i] == target[i]:  # if index equal to target label, record correct classification
                        total_correct += 1

            # average loss = sum of loss over all batches/num of batches
            average_validation_loss = validation_loss / (
                len(self.validation_data_loader.dataset) / self.validation_batch_size)

            # calculate total accuracy for current epoch
            self.accuracy_cur_epoch = 100.0 * total_correct / (len(self.validation_data_loader.dataset))
            # add accuracy for current epoch to list
            self.accuracy.append(self.accuracy_cur_epoch)
            print('\nValidation Epoch {}: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.
                  format(epoch, average_validation_loss, total_correct, len(self.validation_data_loader.dataset),
                         self.accuracy_cur_epoch))
            return average_validation_loss

        'Actual Code starts here, code above are local methods'

        print("\nStarting training of simple neural network using NnImg2Num on MNIST dataset from epoch %r\n" % (self.start_epoch+1))
        self.accuracy_cur_epoch = 0
        # Perform training and validation in each iteration(epoch)
        # Start from 1st epoch or the next one from the last saved epoch, i.e. if saved=n, start (n+1)th epoch
        for i in range(self.start_epoch + 1, self.epochs + 1):
            # Training
            start_time = time.time()
            self.train_loss.append(training(i))
            end_time = time.time() - start_time
            self.computation_time.append(end_time)
            print('\t Computation Time: {:.2f} seconds'.format(end_time))

            # Validation
            self.validation_loss.append(validation(i))

            # save checkpoint
            better = self.accuracy_cur_epoch > self.best_accuracy  # check if accuracy in cur epoch is better than saved
            self.best_accuracy = max(self.best_accuracy, self.accuracy_cur_epoch)  # record the best accuracy
            print('Saving checkpoint after completion of epoch {}'.format(i))
            save_checkpoint({'epoch': i,  # save epoch, best accuracy, model parameters, optimizer state, result lists
                             'best_accuracy': self.best_accuracy,
                             'state_dict': self.nn_model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'train_loss': self.train_loss,
                             'validation_loss': self.validation_loss,
                             'accuracy': self.accuracy,
                             'time': self.computation_time,
                             }, better)
            print('Saved, proceeding to next epoch')
            print('------------------------------------------------------------------------------------')

        print('Average computation time over all iterations {:.2f} seconds'.
              format(np.sum(self.computation_time) / self.epochs))

        # Plot loss vs epoch
        plt.figure(1)
        plt.plot(self.epoch_num, self.train_loss, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Loss')
        plt.plot(self.epoch_num, self.validation_loss, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Loss')
        plt.ylabel('Loss', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)
        title = 'Loss vs Epochs using simple network using Pytorch nn Package, Loss_fn: MSELoss, Optimizer: SGD ' \
                '(learning rate %r) ' % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

        # Plot time vs epoch
        plt.figure(2)
        plt.plot(self.epoch_num, self.computation_time, color='red', linestyle='solid', linewidth='2.0',
                 marker='o', markerfacecolor='red', markersize='5', label='Training Time per epoch')
        plt.ylabel('Computation Time (in seconds)', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)
        title = 'Computation Time vs Epochs using simple network using Pytorch nn Package, Loss_fn: MSELoss, ' \
                'Optimizer: SGD (learning rate %r) ' % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

        # Test forward method
        # label = self.forward(torch.squeeze(self.validation_data_loader.dataset[574][0],0))
        # print(label, self.train_data_loader.dataset[574][1])

    def forward(self, img: torch.ByteTensor):
        """

        This method takes an image from the MNIST dataset and predict the label of the image using the trained neural
        network model built using Pytorch's nn package
        :param img: 28x28 ByteTensor
        :return: [int] predicted label(class)
        """

        img_3D = torch.unsqueeze(img, 0)
        img_4D = Variable(torch.unsqueeze(img_3D, 0))  # Wrap the input into Variable
        # print(img)
        self.nn_model.eval()  # Sets the module in evaluation mode
        output = self.nn_model(img_4D.view(1, self.size1D))  # Forward pass using trained model
        value, pred_label = torch.max(output, 1)  # get index of max value among output class
        return pred_label
