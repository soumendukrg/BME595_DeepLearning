import torch
from torchvision import transforms
from torch.autograd import Variable
import os
import sys
import cv2
import argparse

# Create an ArgumentParser object which will obtain arguments from command line
parser = argparse.ArgumentParser(description="Testing fine-tuned AlexNet for classifying continuous frames from webcam")
parser.add_argument('--model', type=str, help='path to directory where trained model is saved, should be same as the'
                                              ' argument for --save when calling training.py')
# parse_args returns an object with attributes as defined in the add_argument. The ArgumentParser parses command line
# arguments from sys.argv, converts to appropriate type and takes defined action (default: 'store')
args = parser.parse_args()
# We need to do this and then import from train.py as we don't want to provide arguments for train.py
sys.argv = [sys.argv[0]]

from train import AlexNet


class TestModel:
    def __init__(self):
        """
        The defined model AlexNet in train.py is instantiated, state_dict is loaded frm saved model from path args.model
        and copied into model instance
        """
        # Instantiate AlexNet model
        self.model = AlexNet()

        # Load saved model (trained on tiny imagenet)
        filename = 'alexnet_best_model.pth.tar'  # Filename to load checkpoint from
        load_chkpt_file = os.path.join(args.model, filename)  # Path of saved model file
        if os.path.isfile(load_chkpt_file):  # check if provided file exists
            print('\nLoading saved model from file: {}\n'.format(load_chkpt_file))
            chkpt = torch.load(load_chkpt_file)
            # load state of last training
            start_epoch = chkpt['epoch']
            best_accuracy = chkpt['best_accuracy']

            # load model parameters (these are trained parameters)
            self.model.load_state_dict(chkpt['state_dict'])

            # load class names
            self.class_names = chkpt['numeric_class_names']
            self.tiny_class = chkpt['tiny_class']

            print('Completed loading from file: {}, \n(training stopped after epoch {}, best validation accuracy till '
                  'now {:.2f})'.format(load_chkpt_file, start_epoch, best_accuracy))
        else:
            print('\nNo saved model found, no checkpoint to load from, please train model before testing\n')
            sys.exit(0)

    def forward(self, img: torch.ByteTensor):
        """
        This method takes an image tensor and predict the label of the image using the trained neural network
        :param img: 3x224x224 ByteTensor
        :return: [str] predicted label string from tiny_class dict
        """
        # The network expects a batch input, i.e. in form 1x3x224x224, hence we need to add dummy batch dimension. The
        # network works on FloatTensor, hence the conversion inside
        input_image = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        input_image = Variable(input_image)  # Wrap the input into Variable

        # Sets the module in evaluation mode, used as Dropout layer is present
        self.model.eval()

        output = self.model(input_image)  # Do forward pass using trained model to get output
        _, pred_label = torch.max(output, 1)  # get index of max value among output class (200x1)

        # pred_label is the number representing class among 0 to 199, class_names[pred_label.data[0]] is the numerical
        # class label like n02124075, tiny_class[...] represents the actual string label
        label = self.tiny_class[self.class_names[pred_label.data[0]]]
        return label

    def cam(self, idx=0):
        """
        This method is used to fetch images from the camera, it starts the video capture from the webcam, frame by frame
        It displays the video and exits upon keyboard input 'q' or 'Q'
        By default, if you have a single camera, camera index = 0
        :param idx: int
        """
        print("\nStarting Webcam to identify live video frames using AlexNet trained on Tiny ImageNet dataset\n")

        def preprocess(image):
            """
            Local method: Rescale to 256x256x3, crop to 224x224x3, transform to tensor and normalize image
            :param image: image frame captured by webcam
            :return: preprocessed 3x224x224 torch.ByteTensor
            """
            # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_transform = transforms.Compose([transforms.ToPILImage(),    # convert from numpy.ndarray to PIL Image
                                                transforms.Scale(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize])
            return img_transform(image)

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
                # Image Pre-Processing before sending to forward method
                norm_image_tensor = preprocess(frame)

                # Send Image to Pre Trained AlexNet model for prediction of class
                pred_class = self.forward(norm_image_tensor)

                # Display pred class on image
                cv2.putText(frame, pred_class, (250, 50), font, 2, (255, 255, 100), 5, cv2.LINE_AA)
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


if __name__ == '__main__':
    mod = TestModel()
    mod.cam()
