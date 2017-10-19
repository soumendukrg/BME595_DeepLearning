import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import shutil
import os
import argparse
import cv2

# Create an ArgumentParser object which will obtain arguments from command line
parser = argparse.ArgumentParser(description="Fine Tuning pre-trained AlexNet for classifying Tiny ImageNet dataset")
parser.add_argument('--data', type=str, help='path to directory where tiny imagenet dataset is present')
parser.add_argument('--save', type=str, help='path to directory to save trained model after completion of training')
# parse_args returns an object with attributes as defined in the add_argument. The ArgumentParser parses command line
# arguments from sys.argv, converts to appropriate type and takes defined action (default: 'store')
args = parser.parse_args()


class AlexNet(nn.Module):
    def __init__(self):
        """
        Model Definition
        """
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # performs ReLU operation on the conv layer ouput in place
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 200)
        )

    def forward(self, inp):
        """
        Defines the forward computation performed at every call by defined AlexNet network
        """
        out = self.features(inp)
        out = out.view(out.size(0), -1)  # linearized the output of the module 'features'
        out = self.classifier(out)
        out = F.softmax(out)  # apply softmax activation function on the output of the module 'classifier'
        return out


class TrainModel:
    def __init__(self):
        """
        Initialize pretrained AlexNet network, hyper-parameters, and Tiny ImageNet dataloaders
        """

        # ---------- DATA Setup Phase --------- #

        print("\n\n# ---------- DATA Setup Phase --------- #")
        print("Creating separate folders for each class in validation data and storing images belonging "
              "to each class in corresponding folder")
        print("Completed......................")

        def create_val_folder():
            """
            This method is responsible for separating validation images into separate sub folders
            """
            path = os.path.join(args.data, 'val/images')  # path where validation data is present now
            filename = os.path.join(args.data, 'val/val_annotations.txt')  # file where image2class mapping is present
            fp = open(filename, "r")  # open file in read mode
            data = fp.readlines()  # read line by line

            # Create a dictionary with image names as key and corresponding classes as values
            val_img_dict = {}
            for line in data:
                words = line.split("\t")
                val_img_dict[words[0]] = words[1]
            fp.close()

            # Create folder if not present, and move image into proper folder
            for img, folder in val_img_dict.items():
                newpath = (os.path.join(path, folder))
                if not os.path.exists(newpath):  # check if folder exists
                    os.makedirs(newpath)

                if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
                    os.rename(os.path.join(path, img), os.path.join(newpath, img))

        create_val_folder()  # Call method to create validation image folders

        # ---------- DATALOADER Setup Phase --------- #

        'Create TinyImage Dataset using ImageFolder dataset, perform data augmentation, transform from PIL Image ' \
            'to Tensor, normalize and enable shuffling'

        print("\n\n# ---------- DATALOADER Setup Phase --------- #")
        print("Creating Train and Validation Data Loaders")
        print("Completed......................")

        def class_extractor(class_list):
            """
            Create a dictionary of labels from the file words.txt. large_class_dict stores all labels for full ImageNet
            dataset. tiny_class_dict consists of only the 200 classes for tiny imagenet dataset.
            :param class_list: list of numerical class names like n02124075, n04067472, n04540053, n04099969, etc.
            """
            filename = os.path.join(args.data, 'words.txt')
            fp = open(filename, "r")
            data = fp.readlines()

            # Create a dictionary with numerical class names as key and corresponding label string as values
            large_class_dict = {}
            for line in data:
                words = line.split("\t")
                super_label = words[1].split(",")
                large_class_dict[words[0]] = super_label[0].rstrip()  # store only the first string before ',' in dict
            fp.close()

            # Create a small dictionary with only 200 classes by comparing with each element of the larger dictionary
            tiny_class_dict = {}  # smaller dictionary for the classes of tiny imagenet dataset
            for small_label in class_list:
                for k, v in large_class_dict.items():  # search through the whole dict until found
                    if small_label == k:
                        tiny_class_dict[k] = v
                        continue

            return tiny_class_dict

        # Batch Sizes for dataloaders
        self.train_batch_size = 100  # total 500*200 images, 1000 batches of 100 images each
        self.validation_batch_size = 10  # total 10000 images, 10 batches of 1000 images each

        train_root = os.path.join(args.data, 'train')  # this is path to training images folder
        validation_root = os.path.join(args.data, 'val/images')  # this is path to validation images folder

        # The numbers are the mean and std provided in PyTorch documentation to be used for models pretrained on
        # ImageNet data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Create training dataset after applying data augmentation on images
        train_data = datasets.ImageFolder(train_root,
                                          transform=transforms.Compose([transforms.RandomSizedCrop(224),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        normalize]))
        # Create validation dataset after resizing images
        validation_data = datasets.ImageFolder(validation_root,
                                               transform=transforms.Compose([transforms.Scale(256),
                                                                             transforms.CenterCrop(224),
                                                                             transforms.ToTensor(),
                                                                             normalize]))
        # Create training dataloader
        self.train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.train_batch_size, shuffle=True,
                                                             num_workers=5)
        # Create validation dataloader
        self.validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                                  batch_size=self.validation_batch_size,
                                                                  shuffle=False, num_workers=5)

        # list of class names, each class name is the name of the parent folder of the images of that class
        self.class_names = train_data.classes
        self.num_classes = len(self.class_names)
        self.tiny_class = class_extractor(self.class_names)  # create dict of label string for each of 200 classes

        # ---------- MODEL Setup Phase --------- #
        print("\n\n# ---------- MODEL Setup Phase --------- #")
        print("Download pretrained alexnet from torchvision.models, & copy weights to my model except last layer")
        print("Completed......................")

        pretrained_alexnet = models.alexnet(pretrained=True)  # Download pretrained AlexNet model frm torchvision.models

        torch.manual_seed(1)  # Seed generation for random parameter (theta) generation, guarantees same parameters
        self.model = AlexNet()  # Instantiate my model with all layers same as AlexNet except last layer

        # Copying weights from pretrained model to my model for all layers except the last linear layer
        for i, j in zip(self.model.modules(), pretrained_alexnet.modules()):  # iterate over both models
            if not list(i.children()):
                if len(i.state_dict()) > 0:  # copy weights only for the convolution and linear layers
                    if i.weight.size() == j.weight.size():  # this helps to prevent copying of weights of last layer
                        i.weight.data = j.weight.data
                        i.bias.data = j.bias.data

        # Freeze the weights of all the layers of the new model (with pre-trained weights from pre-trained model) except
        # the last linear layer
        for param in self.model.parameters():
            param.requires_grad = False
        # We need to set enable gradient calculation for the last layer, as it has been set to False earlier
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

        # Hyper-parameters for Training
        self.learning_rate = 0.001  # learning rate for optimizer
        self.epochs = 50  # no of times training and validation to be performed on network

        # Set loss function as Cross Entropy Loss
        self.loss_fn = nn.CrossEntropyLoss()  # By default, the losses are averaged over observations for each minibatch

        # We use Adam as the optimization algorithm. The first argument to the Adam constructor tells the
        # optimizer which Variables it should update. We only optimize the parameters of the last layer as we are only
        # training the last layer
        self.optimizer = torch.optim.Adam(self.model.classifier[6].parameters(), lr=self.learning_rate)

        # Initialize list of epochs for plotting
        self.epoch_num = range(1, self.epochs + 1)

        # Load model from the latest saved checkpoint and Resume training
        filename = 'alexnet_model_checkpoint.pth.tar'  # Filename to load checkpoint from
        load_chkpt_file = os.path.join(args.save, filename)  # Path of saved model file
        if os.path.isfile(load_chkpt_file):  # check if provided file exists
            print('\nLoading from saved model checkpoint file: {}\n'.format(load_chkpt_file))
            chkpt = torch.load(load_chkpt_file)
            # load state of la  st training
            self.start_epoch = chkpt['epoch']
            self.best_accuracy = chkpt['best_accuracy']

            # load model parameters
            self.model.load_state_dict(chkpt['state_dict'])
            self.optimizer.load_state_dict(chkpt['optimizer'])

            # load result parameters
            self.train_loss = chkpt['train_loss']
            self.validation_loss = chkpt['validation_loss']
            self.train_accuracy = chkpt['train_accuracy']
            self.validation_accuracy = chkpt['validation_accuracy']
            self.computation_time = chkpt['time']
            print('Completed loading model from checkpoint file: {}, \n(last saved epoch {}, best validation accuracy '
                  '{:.2f})'.format(load_chkpt_file, self.start_epoch, self.best_accuracy))
        else:
            print('\nNo saved model found, no checkpoint to load from\n')
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
        This method trains the network model using torch nn package using Tiny ImageNet dataset.
        There're two main methods: training and validation.
        These two methods are called for a predefined no of epochs.
        training: The model is trained by doing forward pass by passing data to model, finding average batch loss,
            doing backward pass to compute gradients only for last layer, and updating model parameters (theta) in cycle
            for all batches of training data. Final avg training loss and training accuracy is found at end of epoch.
        validation: Once the network is trained once, validation data obtained from the validation data loader is fed to
            the network and the output is compared with the target labels from dataset to find out loss and accuracy.
        :return: nil
        """

        def save_checkpoint(state, better, file=os.path.join(args.save, 'alexnet_model_checkpoint.pth.tar')):
            """
            Local method: This method will save a checkpoint of model parameters and results at end of each epoch
            :param state: dictionary of parameters to save
            :param better: True if current epoch accuracy is better than last saved best accuracy
            :param file: name of file where checkpoint is to be saved
            """
            torch.save(state, file)  # save model state in checkpoint file
            if better:  # if current state is better than all other previous state, save in best model file
                shutil.copyfile(file, os.path.join(args.save, 'alexnet_best_model.pth.tar'))

        def training(epoch: int):
            """
            Local method: Training of model
            :param epoch: number indicating current epoch
            :return: average training loss for current epoch
            """

            def onehot_training():
                """
                USE THIS ONLY FOR MSE Loss
                Local method: This method performs onehot encoding for training labels
                :return: encoded labels for the whole batch of data
                """
                labels_onehot = torch.zeros(self.train_batch_size, self.num_classes)  # initialize with all zero
                for i in range(self.train_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            self.model.train()  # Sets the module in training mode, required for Dropout layer in the model
            training_loss = 0  # initialize total training loss for each epoch
            total_correct = 0  # no of correct classifications in current epoch

            for batch_id, (data, target) in enumerate(self.train_data_loader):
                # Wrap data and target in Variable, no gradient required for target
                data, target = Variable(data), Variable(target, requires_grad=False)

                # We use the optimizer to zero all the gradients for the variables (theta/weights of the model)
                # before the backward pass                                                                             
                self.optimizer.zero_grad()

                # Feed forward pass: Passing current batch of data to the AlexNet model
                output = self.model(data)

                batch_loss = self.loss_fn(output, target)  # compute average Cross Entropy loss for current batch
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

            print("\nAverage Training loss: {:.6f}\t Accuracy: {}/{} ({:.2f}%)".
                  format(average_training_loss, total_correct, len(self.train_data_loader.dataset),
                         self.training_accuracy_cur_epoch))

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
                :param: input is a batch of labels from validation dataset
                :return: encoded labels for the whole batch of data
                """
                labels_onehot = torch.zeros(self.validation_batch_size, self.num_classes)  # initialize labels with 0
                for i in range(self.validation_batch_size):  # loop through all images in batch
                    labels_onehot[i][target[i]] = 1  # make index=1 for col=target label, rest 0
                return labels_onehot

            self.model.eval()  # Sets the module in evaluation mode
            validation_loss = 0  # initialize total validation loss for whole validation dataset for current epoch
            total_correct = 0  # no of correct classifications in current epoch

            for data, target in self.validation_data_loader:
                # Wrap data and target in Variable, no gradient required for target
                data, target = Variable(data), Variable(target, requires_grad=False)

                # Feed forward pass: Passing current batch of data to the AlexNet model
                output = self.model(data)

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

        if self.epochs != self.start_epoch:
            print("\nStarting training of Pre Trained AlexNet network Tiny ImageNet dataset from epoch %r\n" % (
                self.start_epoch + 1))

            # Initialize accuracy variables for this epoch
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
                # check if accuracy in cur epoch is better than saved
                better = self.validation_accuracy_cur_epoch > self.best_accuracy
                self.best_accuracy = max(self.best_accuracy,
                                         self.validation_accuracy_cur_epoch)  # record the best accuracy
                print('Saving model checkpoint after completion of epoch {}'.format(i))
                save_checkpoint(
                    {'epoch': i,  # save epoch, best accuracy, model parameters, optimizer state, result lists
                     'best_accuracy': self.best_accuracy,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'train_loss': self.train_loss,
                     'validation_loss': self.validation_loss,
                     'train_accuracy': self.train_accuracy,
                     'validation_accuracy': self.validation_accuracy,
                     'time': self.computation_time,
                     'numeric_class_names': self.class_names,
                     'tiny_class': self.tiny_class,
                     }, better)
                print('Saved, proceeding to next epoch')
                print('------------------------------------------------------------------------------------')
        else:
            print("\nTraining already completed, if you want to train more, increase self.epochs\n")

        print('Average computation time over all iterations {:.2f} seconds\n'.
              format(np.sum(self.computation_time) / self.epochs))

        # Plot loss vs epoch
        plt.figure(1)
        plt.plot(self.epoch_num, self.train_loss, color='red', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='red', markersize='5', label='Training Loss')
        plt.plot(self.epoch_num, self.validation_loss, color='blue', linestyle='solid', linewidth='2.0',
                 marker='D', markerfacecolor='blue', markersize='5', label='Validation Loss')
        plt.ylabel('Loss', fontsize=24)
        plt.xlabel('Epochs', fontsize=24)

        title = 'Loss vs Epochs using AlexNet model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r) ' \
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

        title = 'Computation Time vs Epochs using AlexNet model, Loss_fn: CrossEntropyLoss, Optimizer: Adam ' \
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

        title = 'Accuracy vs Epochs using AlexNet model, Loss_fn: CrossEntropyLoss, Optimizer: Adam (learning rate %r)' \
                % self.learning_rate
        plt.title(title, fontsize=24)
        plt.legend(fontsize=24)
        plt.grid(True)
        plt.show()

        # TEST MODEL AT END OF TRAINING
        # for i in range(0, 1000, 50):
        #     label = self.tiny_class[self.class_names[self.validation_data_loader.dataset[i][1]]]
        #     print("actual class %r" % label)
        #     self.view(self.validation_data_loader.dataset[i][0], label)

    """
    THE FOLLOWING TWO METHODS ARE NOT REQUIRED FOR HOMEWORK, THEY WERE USED FOR TESTING THE MODEL AFTER TRAINING
    """
    def forward(self, img: torch.ByteTensor):
        """

        This method takes an image tensor and predict the label of the image using the trained neural network
        :param img: 3x224x224 ByteTensor
        :return: [int] predicted label(class)
        """
        # The network expects a batch input, i.e. in form 1x3x224x224, hence we need to add dummy batch dimension. The
        # network works on FloatTensor, hence the conversion inside
        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)  # Wrap the input into Variable

        # Sets the module in evaluation mode, used as Dropout layer is present
        self.model.eval()

        output = self.model(input_image)  # Do forward pass using trained model to get output
        value, pred_label = torch.max(output, 1)  # get index of max value among output class (200x1)

        # pred_label is the number representing class among 0 to 199, class_names[pred_label.data[0]] is the numerical
        # class label like n02124075, tiny_class[...] represents the actual string label
        label = self.tiny_class[self.class_names[pred_label.data[0]]]
        return label

    def view(self, img: torch.ByteTensor, label):
        """
        This method will take a image, display the image and its predicted label by the trained network
        :param img: input image, 3x224x224 torch.ByteTensor
        :param label: label string indicating name of class
        :return:nil
        """
        pred_class = self.forward(img)  # Obtain the predicted class of the input image from the network

        img = img.numpy()  # Convert to numpy
        img = np.transpose(img, (1, 2, 0))  # Transpose from CxHxW to HxWxC (3x224x224 to 224x224x3) for display

        # Un-normalize input image, as during download of the dataset we normalized image tensors
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean

        # As the images are very small, we want to view them in a magnified form. Hence, we create a large window
        cv2.namedWindow(pred_class, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pred_class, 640, 480)  # The viewing height and width of the window is modified
        cv2.imshow(pred_class, img)  # Display the image and its predicted class by the network
        cv2.waitKey(0)  # Waiting after display
        cv2.destroyAllWindows()  # Destroy the window


if __name__ == '__main__':
    a = TrainModel()
    a.train()
