#Main function for Assignment for Image Convolution
from conv import Conv2D
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import time


imgToTensor = transforms.Compose([transforms.ToTensor()])   #Convert PIL.Image to tensor

num_images = 2
#Load Test Images and convert to 3D Float Tensor
input_image0 = Image.open("img0.jpg")
input_image1 = Image.open("img1.jpg")

#input_image2 = Image.open("img2.jpg")
#input_image3 = Image.open("img3.jpg")
input_img_tensor = (imgToTensor(input_image0),imgToTensor(input_image1))
#input_img_tensor = (imgToTensor(input_image0),imgToTensor(input_image1),imgToTensor(input_image2),imgToTensor(input_image3))

computation_time = np.zeros(10) #Variable declaration for Part B, ones less value than 11 as last value will be extrapolated
num_of_ops_partc = np.zeros(5)  #Variable declaration for Part C

#Steps for Part A to C for two input image tensors
for j in range(num_images):
    print("\n-------------------------IMAGE %r (%r)--------------------------\n\n"%(j, input_img_tensor[j].size()))
    print("---------------------PART A------------------------")    
    """PART A"""
    """TASK 1"""
    #Initialize Conv2D
    conv2d = Conv2D(in_channel=3, o_channel=1, kernel_size=3, stride=1, mode ='known')
    [num_of_ops, output_img_tensor] = conv2d.forward(input_img_tensor[j])   #Call forward function with input image tensor
    num_ops = '\nThe total number of operations (multiplications and additions) for Task1 for Image No %d is %d'%(j,num_of_ops)
    print (num_ops)
    output_img_array = output_img_tensor.numpy()    #Convert from Float Tensor to numpy.array
    conv2d.normalize(j,output_img_array)              #Normalize and Save as grayscale image
    
    """TASK 2"""
    #Initialize Conv2D
    conv2d = Conv2D(in_channel=3, o_channel=2, kernel_size=5, stride=1, mode ='known')
    [num_of_ops, output_img_tensor] = conv2d.forward(input_img_tensor[j])   #Call forward function with input image tensor
    num_ops = '\nThe total number of operations (multiplications and additions for Task2 for Image No %d is %d'%(j,num_of_ops)
    print (num_ops)
    output_img_array = output_img_tensor.numpy()    #Convert from Float Tensor to numpy.array
    conv2d.normalize(j,output_img_array)   
    
    """TASK 3"""
    #Initialize Conv2D
    conv2d = Conv2D(in_channel=3, o_channel=3, kernel_size=3, stride=2, mode ='known')
    [num_of_ops, output_img_tensor] = conv2d.forward(input_img_tensor[j])   #Call forward function with input image tensor
    num_ops = '\nThe total number of operations (multiplications and additions for Task3 for Image No %d is %d'%(j,num_of_ops)
    print (num_ops)
    output_img_array = output_img_tensor.numpy()    #Convert from Float Tensor to numpy.array
    conv2d.normalize(j,output_img_array)
    
    
    #### PART B ###
    #Initialize Conv2D
    print("\n------------------Part B--------------------\n")
    for i in range(0,10):   #Only do till o_channel = 512, as for o_channel = 1024, total time taken increases beyond 1 hour
        conv2d = Conv2D(in_channel=3, o_channel=2**i, kernel_size=3, stride=1, mode ='rand')
        start_time = time.time()
        [num_of_ops, output_img_tensor] = conv2d.forward(input_img_tensor[j])   #Call forward function with input image tensor
        computation_time[i] = time.time() - start_time
        print('i = %r, o_channel = %r, number_of_operations = %r, computation_time = %r'%(i,2**i,num_of_ops,computation_time[i]))
            
    i = np.arange(0,10,1)
    
    #Do extrapolation for o_channel = 1024, i = 10 using polynomial curve fitting
    z=np.polyfit(i,computation_time,3)
    func=np.poly1d(z)
    other_i = 10
    other_time = func(other_i)
    
    #Create new dataset using extrapolated value
    i = np.arange(0,11,1)
    final_computation_time = np.hstack((computation_time,other_time))

    #Plot time taken for each forward pass with respect to i
    ll = plt.plot(i,final_computation_time)
    xl = plt.xlabel('i')
    yl = plt.ylabel('Time taken ')
    ttl = plt.title('Time taken for forward pass of convolution vs i')
    grd = plt.grid(True)
    plt.show()
    fig_name = '/home/soumendu/Documents/Python /BME 595/Assignments/PlotB_TimeTaken_vs_i_image%s.jpg'%(j)
    plt.savefig(fig_name, bbox_inches = 'tight')
    plt.close()

    """
    ### PART C ###
    #Initialize Conv2D
    print("\n------------------Part C--------------------\n")
    k_size = [3,5,7,9,11]    #Note computation for kernel size 3 already done
    for i in range(0,5):
        conv2d = Conv2D(in_channel=3, o_channel=2, kernel_size=k_size[i], stride=1, mode ='rand')        
        [num_of_ops_partc[i], output_img_tensor] = conv2d.forward(input_img_tensor[j])   #Call forward function with input image tensor
        print('i = %r, o_channel = %r, kernel_size = %r, number_of_operations = %r'%(i, 2, k_size[i], num_of_ops_partc[i]))
            
    #Plot number of operations =with respect to kernel size
    ll = plt.plot(k_size,num_of_ops_partc)
    xl = plt.xlabel('kernel size')
    yl = plt.ylabel('Number of Operations')
    ttl = plt.title('Number of Operations vs Kernel Size')
    grd = plt.grid(True)
    plt.show()
    fig_name = '/home/soumendu/Documents/Python /BME 595/Assignments/PlotC_NumOps_vs_kernelsize_image%s.jpg'%(j)
    plt.savefig(fig_name, bbox_inches = 'tight')
    plt.close()
    #"""


