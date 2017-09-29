import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil, os
import cv2


class LeNet5(nn.Module):
    def __init__(self):
        """
        Model Definition
        """
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=0)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.l1 = nn.Linear(16 * 5 * 5, 120)
        self.l2 = nn.Linear(120, 84)
        self.l3 = nn.Linear(84, 100)

    def forward(self, x):
        """
        Defines the forward computation performed at every call by defined LeNet5 module
        """
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out


class img2obj:
    def __init__(self):
        """

        Initialize neural network, hyper-parameters, and CIFAR100 data loaders
        """
        # Hyper-parameters for Training
        self.train_batch_size = 125  # total 60000 images, 400 batches of 125 images each
        self.validation_batch_size = 1000  # total 10000 images, 10 batches of 1000 images each
        self.learning_rate = 0.001  # learning rate for optimizer
        self.epochs = 50  # no of times training and validation to be performed on network

        # Image size parameters
        channels = 3  # number of channels of input image
        row = 32  # number of rows of input image
        col = 32  # number of columns of input image
        # The 3D image tensor has to be converted to 1D tensor, each row has to be placed along the column dimension
        # one after another, and the process is repeated for each of the input channels, size1D is the no of columns of
        # the converted 1D tensor
        self.size1D = channels * row * col
        # Output classes, these are arranged alphabetically according to the dataset, with some exceptions which have
        # been modified after observing first 100 images of training dataset
        self.classes = ('apples', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
                        'bottles', 'bowls', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'cans', 'castle',
                        'caterpillar',
                        'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'kangaroo', 'couch',
                        'crocodile', 'cups', 'crab', 'dinosaur', 'elephant', 'dolphin', 'flatfish', 'forest', 'girl',
                        'fox', 'hamster', 'house', 'computer keyboard', 'lamp', 'lawn-mower', 'leopard', 'lion',
                        'lizard',
                        'lobster', 'man', 'maple', 'motorcycle', 'mountain', 'mouse', 'mushrooms', 'oak', 'oranges',
                        'orchids', 'otter', 'palm', 'pears', 'pickup truck', 'pine', 'plain', 'plates', 'poppies',
                        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'roses', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflowers', 'sweet peppers', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulips', 'turtle', 'wardrobe', 'whale', 'willow', 'wolf', 'woman', 'worm'
                        )
        self.labels = len(self.classes)

        # local inline function to normalize images to have 0 mean and 1 std so that network can learn features
        # effectively
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        'Download CIFAR100 dataset (if not already downloaded), perform data augmentation, transform from PIL Image ' \
        'to Tensor, normalize and enable shuffling'
        # Training dataset loader: when called, it will create shuffled train data of train_batch_size
        # ../CIFAR100_data represents root directory where raw data is downloaded to and processed data is loaded from
        self.train_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../CIFAR100_data', train=True, download=True,
                              transform=transforms.Compose(
                                  [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])),
            batch_size=self.train_batch_size, shuffle=True, num_workers=5)

        # Validation dataset: it will create non-shuffled normalized validation data of validation_batch_size
        self.validation_data_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('../CIFAR100_data', train=False,
                              transform=transforms.Compose([transforms.ToTensor(), normalize])),
            batch_size=self.validation_batch_size, shuffle=False, num_workers=5)

        torch.manual_seed(1)  # Seed generation for random parameter (theta) generation, guarantees same theta always
        # Create neural network model and loss function using nn package
        self.nn_model = LeNet5()

        # Set loss function as Cross Entropy Loss
        self.loss_fn = nn.CrossEntropyLoss()  # By default, the losses are averaged over observations for each minibatch

        # We define an Optimizer using the optim package. This will update the theta (weights) of the model.
        # We use Adam as the optimization algorithm. The first argument to the Adam constructor tells the
        # optimizer which Variables it should update, lr specifies the learning rate.
        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.learning_rate, weight_decay=0.0005)

        # Initialize list of epochs for plotting
        self.epoch_num = range(1, self.epochs + 1)

        # Load model from the latest saved checkpoint and Resume training
        load_chkpt_file = 'img2obj_model_checkpoint.pth.tar'  # Filename to load checkpoint from
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
            self.train_accuracy = chkpt['train_accuracy']
            self.validation_accuracy = chkpt['validation_accuracy']
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
            self.train_accuracy = list()
            self.validation_accuracy = list()
            self.computation_time = list()

    def train(self):
        """
        This method trains the network model using torch nn package using CIFAR100 dataset.
        There're two main methods: training and validation.
        These two methods are called for a predefined no of epochs.
        training: The model is trained by doing forward pass by passing data to model, finding average batch loss,
            doing backward pass to compute gradients, and updating model parameters (theta) in cycle for all batches of
            training data. Final average training loss and training accuracy is found at end of epoch.
        validation: Once the network is trained, validation data obtained from the validation data loader is fed to the
            network and the output is compared with the target labels from the dataset to find out loss and accuracy.
        :return: nil
        """

        def save_checkpoint(state, better, file='img2obj_model_checkpoint.pth.tar'):
            """
            Local method: This method will save a checkpoint of model parameters and results at end of each epoch
            :param state: dictionary of parameters to save
            :param better: True if current epoch accuracy is better than last saved best accuracy
            :param file: name of file where checkpoint is to be saved
            """
            torch.save(state, file)  # save model state in checkpoint file
            if better:  # if current state is better than all other previous state, save in best model file
                shutil.copyfile(file, 'img2obj_best_model.pth.tar')

        def training(epoch: int):
            """
            Local method: Training of model
            :param epoch: number indicating current epoch
            :return: average training loss for current epoch
            """

            def onehot_training():
                """
                # USE THIS ONLY FOR MSE Loss
                Local method: This method performs onehot encoding for training labels
                :return: encoded labels for the whole batch of data
                """
                labels_onehot = torch.zeros(self.train_batch_size, self.labels)  # initialize with all zero
                for i in range(self.train_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            self.nn_model.train()  # Sets the module in training mode
            training_loss = 0  # initialize total training loss for each epoch
            total_correct = 0  # no of correct classifications in current epoch

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                # target = onehot_training()  # convert target labels to onehot labels
                # Wrap data and target in Variable, no gradient required for target
                data, target = Variable(data), Variable(target, requires_grad=False)

                # We use the optimizer to zero all the gradients for the variables (theta/weights of the model)
                # before the backward pass                                                                             
                self.optimizer.zero_grad()

                # Feed forward pass: Passing current batch of data to the LeNet5 model
                output = self.nn_model(data)

                batch_loss = self.loss_fn(output, target)  # compute average MSE loss for current batch
                training_loss += batch_loss.data[0]  # add current batch loss to total training loss for current epoch

                # Backward pass: compute gradient of the loss with respect to model parameters (theta)
                batch_loss.backward()

                # Update the model parameters (theta)
                self.optimizer.step()

                value, index = torch.max(output.data, 1)  # get index of max value among output class
                # Compute total no of correct predictions by the trained model in current epoch
                for i in range(0, self.train_batch_size):
                    if index[i] == target.data[i]:  # if index equal to target label, record correct classification
                        total_correct += 1

                print('\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.
                      format(epoch, (batch_id + 1) * self.train_batch_size, len(self.train_data_loader.dataset),
                             100.0 * (batch_id + 1) * self.train_batch_size / len(self.train_data_loader.dataset),
                             batch_loss.data[0]), end="")

            # average loss = sum of loss over all batches/num of batches
            average_training_loss = training_loss / (len(self.train_data_loader.dataset) / self.train_batch_size)

            # calculate total accuracy for the current epoch
            self.training_accuracy_cur_epoch = 100.0 * total_correct / (len(self.train_data_loader.dataset))
            # add accuracy for current epoch to list
            self.train_accuracy.append(self.training_accuracy_cur_epoch)

            print("\t Average Training loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".
                  format(average_training_loss, total_correct, len(self.train_data_loader.dataset),
                         self.training_accuracy_cur_epoch), end="")

            return average_training_loss

        def validation(epoch):
            """
            Local method: Validation of model
            :param epoch: number indicating current epoch
            :return: average validation loss for current epoch
            """

            def onehot_validation(target):
                """
                USE THIS ONLY FOR MSE Loss
                Local method: This method performs onehot encoding for validation labels
                :param: input is a batch of labels from validation dataset of size 1000
                :return: encoded labels for the whole batch of data
                """
                labels_onehot = torch.zeros(self.validation_batch_size, self.labels)  # initialize labels with all zeros
                for i in range(self.validation_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            self.nn_model.eval()  # Sets the module in evaluation mode
            validation_loss = 0  # initialize total validation loss for whole validation dataset for current epoch
            total_correct = 0  # no of correct classifications in current epoch

            for data, target in self.validation_data_loader:
                # target = onehot_validation(target)  # convert target labels to onehot labels
                # wrap data and target in Variable, no gradient required for target
                data, target = Variable(data), Variable(target, requires_grad=False)

                # Feed forward pass: Passing current batch of data to the LeNet5 model
                output = self.nn_model(data)

                batch_loss = self.loss_fn(output, target)  # compute average Cross Entropy loss for current batch
                validation_loss += batch_loss.data[0]  # add current batch loss to total validation loss

                value, index = torch.max(output.data, 1)  # get index of max value among output class
                # Compute total no of correct predictions by the trained model in current epoch
                for i in range(0, self.validation_batch_size):
                    if index[i] == target.data[i]:  # if index equal to target label, record correct classification
                        total_correct += 1

            # average loss = sum of loss over all batches/num of batches
            average_validation_loss = validation_loss / (
                len(self.validation_data_loader.dataset) / self.validation_batch_size)

            # calculate total accuracy for the current epoch
            self.validation_accuracy_cur_epoch = 100.0 * total_correct / (len(self.validation_data_loader.dataset))
            # add accuracy for current epoch to list
            self.validation_accuracy.append(self.validation_accuracy_cur_epoch)

            print('\nValidation Epoch {}: Average loss: {:.6f} \t Accuracy: {}/{} ({:.2f}%)\n'.
                  format(epoch, average_validation_loss, total_correct, len(self.validation_data_loader.dataset),
                         self.validation_accuracy_cur_epoch))
            return average_validation_loss

        'Actual Code starts here, code above are local methods'

        print("\nStarting training of LeNet5 network using img2num on CIFAR100 dataset from epoch %r\n" % (
            self.start_epoch + 1))

        # Initailize accuracy variables for this epoch
        self.training_accuracy_cur_epoch = 0
        self.validation_accuracy_cur_epoch = 0

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
            better = self.validation_accuracy_cur_epoch > self.best_accuracy  # check if accuracy in cur epoch is better than saved
            self.best_accuracy = max(self.best_accuracy, self.validation_accuracy_cur_epoch)  # record the best accuracy
            print('Saving checkpoint after completion of epoch {}'.format(i))
            save_checkpoint({'epoch': i,  # save epoch, best accuracy, model parameters, optimizer state, result lists
                             'best_accuracy': self.best_accuracy,
                             'state_dict': self.nn_model.state_dict(),
                             'optimizer': self.optimizer.state_dict(),
                             'train_loss': self.train_loss,
                             'validation_loss': self.validation_loss,
                             'train_accuracy': self.train_accuracy,
                             'validation_accuracy': self.validation_accuracy,
                             'time': self.computation_time,
                             }, better)
            print('Saved, proceeding to next epoch')

            # TEST YOUR MODEL AT END OF EACH EPOCH
            # print("Press Any key to continue.......................")
            # self.view(self.train_data_loader.dataset[10][0])
            # print("Actual Label %r" % self.classes[self.train_data_loader.dataset[10][1]])
            # time.sleep(3)
            # print("Press Any key to continue.......................")
            # self.view(self.validation_data_loader.dataset[50][0])
            # print("Actual Label %r" % self.classes[self.validation_data_loader.dataset[50][1]])
            # time.sleep(3)

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

        # title = 'Loss vs Epochs using LeNet5 model, Loss_fn: MSELoss, Optimizer: SGD (learning rate %r) ' \
        #         % self.learning_rate

        title = 'Loss vs Epochs using LeNet5 model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r) ' \
                % self.learning_rate
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

        # title = 'Computation Time vs Epochs using LeNet5 model, Loss_fn: MSELoss, Optimizer: SGD (learning rate %r) '\
        #         % self.learning_rate

        title = 'Computation Time vs Epochs using LeNet5 model, Loss_fn: CrossEntropyLoss, Optimizer: Adam ' \
                '(learning rate %r) ' % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

        # Plot accuracy vs epoch
        plt.figure(3)
        plt.plot(self.epoch_num, self.train_accuracy, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Accuracy')
        plt.plot(self.epoch_num, self.validation_accuracy, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Accuracy')
        plt.ylabel('Accuracy', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)

        # title = 'Accuracy vs Epochs using LeNet5 model, Loss_fn: MSELoss, Optimizer: SGD (learning rate %r) ' \
        #         % self.learning_rate

        title = 'Accuracy vs Epochs using LeNet5 model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r)' \
                % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

        # TEST MODEL AT END OF TRAINING
        # for i in range(0, 100):
        #     label = self.classes[self.train_data_loader.dataset[i][1]]
        #     self.view(self.train_data_loader.dataset[i][0], label)

    def forward(self, img: torch.ByteTensor):
        """

        This method takes an image tensor and predict the label of the image using the trained neural network
        :param img: 3x32x32 ByteTensor
        :return: [int] predicted label(class)
        """
        # The network expects a batch input, i.e. in form 1x3x32x32, hence we need to add dummy batch dimension. The
        # network works on FloatTensor, hence the conversion inside
        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)  # Wrap the input into Variable

        # Sets the module in evaluation mode, this is done to ensure that if the model was loaded
        # from a checkpoint, it do not remain in training form during only prediction using forward pass
        self.nn_model.eval()

        output = self.nn_model(input_image)  # Do forward pass using trained model to get output
        value, pred_label = torch.max(output, 1)  # get index of max value among output class (100x1)

        return self.classes[pred_label.data[0]]  # Return string from self.classes whose index matches the pred label

    def view(self, img: torch.ByteTensor):
        """

        This method will take a image, display the image and its predicted label by the trained network
        :param img: input image, 3x32x32 torch.ByteTensor
        :return:nil
        """
        pred_class = self.forward(img)  # Obtain the predicted class of the input image from the network

        # Un-normalize input image, during download of the dataset we normalized image tensors from -0.5 to 0.5 range,
        # so, we will restore the range of data from 0 to 1, so that it can be displayed
        image = img.type(torch.FloatTensor) / 2 + 0.5

        # Convert to numpy format from tensor and transpose from CxHxW to HxWxC for display
        image_numpy = np.transpose(image.numpy(), (1, 2, 0))

        # As the images are very small, we want to view them in a magnified form. Hence, we create a large window
        cv2.namedWindow(pred_class, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pred_class, 640, 480)  # The viewing height and width of the window is modified
        cv2.imshow(pred_class, image_numpy)  # Display the image and its predicted class by the network
        cv2.waitKey(0)  # Waiting after display
        cv2.destroyAllWindows()  # Destroy the window

    def cam(self, idx=0):
        """

        This method is used to fetch images from the camera, it starts the video capture from the webcam, frame by frame
        It displays the video and exits upon keyboard input 'q' or 'Q'
        By default, if you have a single camera, camera index = 0
        :type cam_idx: int
        """

        def preprocess(image):
            """
            Local method: Rescale to 32x32x3 image as model trained on images of this dimension, convert to Tensor
            3x32x32 and normalize
            :param image: image frame captured by webcam
            :return: preprocessed 3x32x32 torch.ByteTensor
            """
            scaled_image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
            # Convert to Tensor and Normalize
            preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                                         std=[0.5, 0.5, 0.5])])
            return preprocess(scaled_image)

        cap_obj = cv2.VideoCapture(idx)  # Create a VideoCapture object with 0 as the argument to start live stream.
        print("\n..............Press q to Quit video capture.............\n")
        font = cv2.FONT_HERSHEY_SIMPLEX  # Set font for text display on video
        # Set default viewing window
        cap_obj.set(3, 1280)
        cap_obj.set(4, 720)

        while True:
            # Capture frame by frame
            read, frame = cap_obj.read()

            if read:  # if successfully read, display video and predicted class by the network
                # cv2.imwrite('Webcam Image Normal.png', frame)   # Saved image to disk
                # Image Pre-Processing before sending to forward method
                norm_image_tensor = preprocess(frame)

                # Send Image to Pre Trained Network for prediction of class
                pred_class = self.forward(norm_image_tensor)

                # Display pred class on image
                cv2.putText(frame, pred_class, (250, 50), font, 2, (255, 200, 100), 5, cv2.LINE_AA)
                cv2.imshow('Webcam Live Video', frame)  # Displaying the frame.

            else:
                print('\nError is reading video frame from the webcam..Exiting..')
                break

            # Extract last byte from the return value of waitKey function as it contains the ASCII values of input
            # from keyboard. Break the loop is user presses 'q' or 'Q'
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        cap_obj.release()  # Releasing the capture
        cv2.destroyAllWindows()  # Closing the window
