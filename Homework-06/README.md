# BME 595 DEEP LEARNING

### Soumendu Kumar Ghosh


## HOMEWORK-06: Neural Networks - CNN in PyTorch
## OBJECTIVE 
**Fine tuning a pre-trained AlexNet model for classifying Tiny ImageNet Dataset**

## Platform and Packages
- **Anaconda** 4.3.27, **Python** 3.6.2, **Pytorch** 0.2.0, **Torchvision** 0.1.9, **OpenCV3** 3.1.0
- Processor: **Intel® Core™ i7Machine-6500U CPU @ 2.50GHz × 4**, RAM: **15.4 GB**
- OS: **Ubuntu 16.04 LTS** 64 bit

## INTRODUCTION
This assignment deals with **transfer learning**. Generally, training an entire Convolutional Network (ConvNet) from scratch (with random initialization) is a cumbersome task, due to two main reasons. The major one is that its rare to have a sufficiently large dataset. Training a large network on a small dataset may lead to overfitting as the network tends to memorize the training dataset and perform very poorly on new data. The second reason is that few people actually have access to fast GPUs and training the large networks on CPUs takes a lot of time. Hence, people pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the pre-trained *ConvNet* either as an initialization or a fixed feature extractor for the purpose of classification new dataset of interest.

Two types of approach exists in the transfer learning domain, described as follows:
- **Finetuning the ConvNet**: The network is initialized with a pretrained network, i.e. the weights are copied from a pretrained network into the new network (provided all the layers are same). This network is then trained as usual.
- **ConvNet as fixed feature extractor**: In this case, a number of layers at the end of the network is replaced with new layers with randomly initialized weights. (**In this homework, only the last fully connected layer of AlexNet is replaced.**) Then, the weights for all of the network except that of the replaced layers are freezed. Finally, the replaced layers are trained on the new dataset.

## DATASET
- [**Tiny Imagenet**](https://tiny-imagenet.herokuapp.com/): This dataset is a small part of the [ImageNet dataset](http://www.image-net.org/). The dataset has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images. Class labels are provided for both training and validation images.

## IMPLEMENTATION
### TRAINING: Fine Tuning pre-trained AlexNet for classifying Tiny ImageNet dataset
The file *[train.py](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py)* implements the following. The finetuning of the model is performed in several stages.

#### Data Setup Phase
The validation images of the tiny imagenet dataset are in a single folder `/tiny-imagenet-200/val/images` and the labels for each image are in the file `/tiny-imagenet-200/val/val_annotations.txt`. The validation folder needs to be organized so that the directory structure is as shown below. The method `create_val_folder()` performs this operation.
```bash
|-val
    |-images
        |-[:class 0] # (name of folder: n01443537)
            |-[:val_img 0]
            |-[:val_img 1]
            |-[:val_img 2]
            ...
        |-[:class 1] # (name of folder: n01629819)
        |-[:class 2] # (name of folder: n01641577)
        ...
            ...
```
#### Dataloader Setup Phase
- The training and validation datasets are created using `torchvision.datasets.ImageFolder`. Adequate data augmentation is performed.
- Dataloaders are created for training and validation datasets.
- The class labels which are also the folder names of the images in train and val folder of **tiny-imagenet-200** are numerical labels like n02124075, n04067472, etc. The actual class label string for each of these numerical labels of the whole ImageNet dataset are in the file `tiny-imagenet-200/words.txt`. A dictionary matching the numerical labels with label string for the 200 classes of the tiny imagenet dataset is created here. 

#### Model Setup Phase
- MODEL: AlexNet [[1](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)]
  - **2D Convolution layer** (`input channel=3, output channel=64, kernel_size=11, stride=4, padding=2`)
  - **ReLU** (`inplace=True`)
  - **2D MaxPool layer**(`kernel_size=3, stride=2`)
  - **2D Convolution layer** (`input channel=64, output channel=192, kernel_size=5, stride=1, padding=2`)
  - **ReLU** (`inplace=True`)
  - **2D MaxPool layer**(`kernel_size=3, stride=2`)
  - **2D Convolution layer** (`input channel=192, output channel=384, kernel_size=3, stride=1, padding=1`)
  - **ReLU** (`inplace=True`)
  - **2D Convolution layer** (`input channel=384, output channel=256, kernel_size=3, stride=1, padding=1`)
  - **ReLU** (`inplace=True`)
  - **2D Convolution layer** (`input channel=256, output channel=256, kernel_size=3, stride=1, padding=1`)
  - **ReLU** (`inplace=True`)
  - **2D MaxPool layer**(`kernel_size=3, stride=2`): The output of this layer is (256,6,6), so it needs to be reshaped to (256 * 6 * 6, 1) before sending to the next layer
  - **Dropout layer**(`p=0.5`)
  - **Fully connected layer**(`input neurons=256 * 6 * 6, output neurons=4096`)
  - **ReLU** (`inplace=True`)
  - **Dropout layer**(`p=0.5`)
  - **Fully connected layer**(`input neurons=4096, output neurons=4096`)
  - **ReLU** (`inplace=True`)
  - **Fully connected layer**(`input neurons=4096, output neurons=200`): Only this layer is different from the pretrained AlexNet network (obtained from *torchvision.model* which had 1000 neurons)
  
- **Softmax** activation function is applied on the output of the network
- The pretrained model **alexnet** from `torchvision.models` is loaded and the parameters for all layers except the last fully connected layer is copied into the new AlexNet model.
- Parameters for all layers except last layer is freezed by setting `requires_grad=False` for each parameter in `model.parameters()`
- **LOSS FUNCTION**: Cross Entropy Loss
- **OPTIMIZER**: *Adam* (only last layer will be optimized)

#### Hyperparameters
|     Type             |  Value  |
|----------------------|---------|
|Training Batch Size   |  100    | 
|Validation Batch Size |   10    |
|Learning Rate for Adam|  0.001  |
|Number of Epochs      |   50    |

#### Training Phase
- In each epoch, the last layer of the network is trained on the training dataset followed by running forward pass on the validation dataset. Loss and Accuracy are calculated.
-  At the end of each epoch, the epoch no, best accuracy, model state dictionary, optimizer dictionary, dictionary of class labels and list of results (training loss, validation loss, validation accuracy, and computation time for training) are saved in the location specified by `args.save`. There can be two uses of the saved model checkpoint:
    - Training can be resumed later from a saved checkpoint.
    - The saved network can be directly used for testing new images.
  
### TESTING: Testing fine-tuned AlexNet for classifying continuous frames from webcam
The file *[test.py](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/test.py)* implements the following.
#### Model Setup Phase
- Instantiate `AlexNet()` model from `train.py`. Load the saved model checkpoint from `args.model`. Copy model state dictionary into the instantiated model and get the dictionary of class labels for displaying label on webcam frame.
#### Testing camera frames
- The video capture from the webcam is started, i.e. frames are captured continuously.
- Each frame is preprocessed (scaled down to 224x224 from 1280x720, converted to tensor and normalized) and sent to the trained model which in turn returns the predicted label to this method.
- Live Video from the webcam is displayed using OpenCV and the predicted label is overlayed on the video.
  
## Running the tests
#### Training: 
The following code shows how to run the script. 
```bash
$ python train.py -h  # -h argument show what arguments should be passed to this file
usage: train.py [-h] [--data DATA] [--save SAVE]
Fine Tuning pre-trained AlexNet for classifying Tiny ImageNet dataset
optional arguments:
  -h, --help   show this help message and exit
  --data DATA  path to directory where tiny imagenet dataset is present
  --save SAVE  path to directory to save trained model after completion of
               training
               
# Finally run the file with proper arguments
$ python train.py --data /tiny/imagenet/dir/ --save /dir/to/save/model/
```

#### Testing:
The following code shows how to run the script.
```bash
$ python test.py -h  # -h argument show what arguments should be passed to this file
usage: test.py [-h] [--model MODEL]
Testing fine-tuned AlexNet for classifying continuous frames from webcam
optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  path to directory where trained model is saved, should be
                 same as the argument for --save when calling training.py
               
# Finally run the file with proper arguments
$ python3 test.py --model /dir/containing/model/ # this path should be same as /dir/to/save/model/ 
                                             # as the trained model in train.py was saved in this location
```

## Results and Observations
### Output of `train.py`
#### Loss vs Epoch
- Result after 25 epochs
![Plot1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/Results/LossvsEpoch_Alexnet.png)

#### Time vs Epoch
- Result after 25 epochs
![Plot2](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/Results/TimevsEpoch_Alexnet.png)

#### Accuracy vs Epoch
- Result after 25 epochs
![Plot3](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/Results/AccuracyvsEpoch_Alexnet.png)

#### **Table**: Results of AlexNet on Tiny ImageNet

|  Model      |   Epoch   |  Training Accuracy| Validation Accuracy|
|-------------|-----------|-------------------|--------------------|
|AlexNet Model|     25    |      35.34%       |     45.26%         |

- The validation accuracy is better than the training accuracy probably as the no of validation images (10000) is much less than the training images (100000), so there is more chance of getting correct predictions among less no. of images.

### Output of `test.py`
-  A snapshot of an image frame captured using the webcam which is *correctly identified* by the trained network is shown here.
![Camview](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/Results/Correctly_Identified_Frame_AlexNet.png)

## REFERENCES
[1] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
