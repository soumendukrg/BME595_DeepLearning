# **BME 595 DEEP LEARNING**

### **Soumendu Kumar Ghosh**


# **HOMEWORK-01:** 
*Implementation of 2D Convolution*

The results of different parts of the howework are described in this file.

##### Platform and Packages used:
- Anaconda 4.3.25, Python 3.6.2, Pytorch, Torchvision (All updated)
- Machine: Intel core i7 (4 cores), 16GB RAM
- OS: Ubuntu 14.04 in Virtual Box with 8 GB RAM and 2 cores allocated, 20GB Virtual Disk.

## **API**

### Python
  - The class Conv2D is defined in the file *conv.py*.
  - The `__init__()` and the `forward()` functions have been defined in the same.
  - An additional function called `normalize(img_no,output_img_array)` has been defined in *conv.py* for normalizing and saving the output image. The output image tensor (type: _3D FloatTensor_) is converted in *main.py* to a numpy array and then passed as the argument of the normalize function. Each channel of the output image array is saved as a separate grayscale image in the function.

#### Details about *main.py*
  - Both the test images are read. _img0.jpg_ (1280x720) and _img1.jpg_ (1920x1080) are the two corresponding test images in the assignment folder.
  - Image 0
![Image0](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/img0.jpg)
  - Image 1
![Image1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/img1.jpg)
  - The test images are transformed from PIL.Image format to torch.FloatTensor format.
  - Code for Part A through C are present in this file. Please note that **either Part B or Part C can be run at a time**. Please comment out code section for Part C when you want to run Part B and viceversa. Part A has no such limitations.
  - A _for_ loop has been used to run all the computations for all the images in one pass of the program. If you want to use other images than those provided, you can either replace the argument of image read instructions or you can also append newly read images into the tensor `input_img_tensor`. Make sure to transform yours images to torch.FloatTensor before using. The variable `num_images` is used for looping through the images.

### C

#### Details of the C API
  - The function `long double c_conv(args)` has been defined in *main.py*. The data type has been changed from _int_ to _long double_ as the number of operations which is returned by this function is of type _long double_.
  - The function declaration is shown below:
  
  ```python
  long double c_conv(int in_channel, long int o_channel, int kernel_size, int stride, float ***input_image);
  ```
  - 3D array was created with random values between 0 to 255 to replicate images with dimensions 1280x780x3 and 1980x720x3 in two different execution instances of the program. The submitted code has the size as 1280x720x3. Change the variables *rows* and *columns* in both the main function and the c_conv function to try other image dimensions.
  - A new argument `float ***input_image` was added in the c_conv function as the image array was created in the main function. The pointer to the array was passed as the argument.
  - Kernel of size=3x3x3 was created in c_conv and used for all output channels (2^i).
  
## Running the tests in Python

Run the following code (without $ sign) in the terminal for Part A to C (remember to comment out PartB or C as required)

```
$ python main.py
```

## **RESULTS**

### PART A
 
The directory structure for the 12 output images for the two images is as below:
- results_PartA
  - Image 0
    - 6 Output Images (all images are named with task no and kernel no used).
    - TASK 1: As evident from the output images, kernel k1 gives the horizontal edge detector, hence the horizontal edges are detected in the output image for task 1. Before the convolution operation, output image tensor of size equal to input image tensor size has been initialized with all zeros. Hence, the final output image size is same as input image sizse after the convolution. The convolution operation modifies the values of output tensor from `row = roi_offset to img_height-roi_offset` and `column = roi_offset to img_width-roi_offset` where roi_offset = 1 as kernel size = 2. 
    - TASK 2: Kernel k4 and k5 gives horizontal and vertical edge detector respectively. However, as kernel size has increased, the sharpness of the output images for task 2 has decreased with respect to task 1. Basically, the edges in the image becomes thicker.
    - TASK 3: K1, K2, and K3 are used here. As the stride = 2 in this task, the size of the output image reduces in both the dimensions by a factor of 2. K1 gives horizontal edge detector, K2 is vertical edge detector and K3 is smoothing filter/averaging filter.
    
  - Image 1
    - Same as above

### PART B

In this part, the number of output channels is varied as 2^i (i=0,1,2,...,10). Since mode is 'rand', a random kernel (3x3x3) is generated in *conv.py* and used for convolution to generate 2^i no of output channels. The total time taken for each of the *forward()* pass was used to generate plot as a function of i.

The directory structure for the two images is as below:
- results_PartB
  - Image 0
    - PlotB_TimeTaken_vs_i_image0.jpeg
    - ![PlotB_0](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/results_PartB/Image%200/PlotB_TimeTaken_vs_i_image0.jpeg)
  - Image 1
    - PlotB_TimeTaken_vs_i_image1.jpeg
    - ![PlotB_1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/results_PartB/Image%201/PlotB_TimeTaken_vs_i_image1.jpeg)
    
It can be observed from both of the plots that the computation time rises exponentially as number of output channels is increased (_the number of output channels also rises exponentially, 2^i_).

###### NOTE for Part B:
The system was taking longer than one hour for no of output channels = 1024 for the smaller image (image 0) and for no of output channels = 512 and 1024 for the larger image(image 1). Computation time for these values were found out by extrapolation using polynomial fitting, and plotted subsequently along with previously obtained values. 

### PART C

In this part, the number of output channels was 2 but the kernel size was (3,5,7,9,11). Since mode is 'rand', a random kernel (3 x kernelsize x kernelsize) was generated in *conv.py* and used for convolution to generate 2 output channels in each case. The total number of operations returned by _forward()_ was used to generate plot as a function of kernelsize.

The directory structure for the two images is as below:
- results_PartC
  - Image 0
    - PlotC_NumOps_vs_kernelsize_image0.jpeg
    - ![PlotC_0](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/results_PartC/Image%200/PlotC_NumOps_vs_kernelsize_image0.jpeg)
  - Image 1
    - PlotC_NumOps_vs_kernelsize_image1.jpeg
    - ![PlotC_1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/results_PartC/Image%201/PlotC_NumOps_vs_kernelsize_image1.jpeg)

It can be observed from both of the plots that the number of operation rises almost linearly as kernel size increaes (by 2).

## Running the tests in C

Run the following code (without $ sign) in the terminal for Part D

```
$ gcc main.c -lm -o main.out
$ ./main.out
```

### PART D

As mentioned earlier, input image array was generated with random values in range (0,255) to replicate image. Convolution was performed in C using the c_conv function. This function returned number of operations needed for convolution and total computation time to the main function from which they were printed and noted down. The values of time taken for different number of output channels were then imported to a python file, from which they were plotted with respect to i.

The directory structure for the two images is as below:
- results_PartD
  - Image 0
    - PlotD_TimeTaken_vs_i_image0.jpeg
    - ![PlotD_0](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/results_PartD/Image0/PlotD_TimeTaken_vs_i_image0.jpeg)
  - Image 1
    - PlotD_TimeTaken_vs_i_image1.jpeg
    - ![PlotD_1](https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-01/results_PartD/Image1/PlotD_TimeTaken_vs_i_image1.jpeg)
    
It can be observed from both of the plots that the computation time rises exponentially as number of output channels is increased (_the number of output channels also rises exponentially, 2^i_).

###### NOTE for Part D:
In case of the larger image (1920x1080x3), the program was killed by the OS after number of output channels increased to 1024. This happened because of lack of virtual memory, as I am using virtual box. The last value for this were extrapolated as before.
