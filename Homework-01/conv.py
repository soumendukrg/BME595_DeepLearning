from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class Conv2D:
    """Class for 2D Convolution"""
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        
        #Define the known kernels
        self.k1 = torch.Tensor([[-1, -1, -1], [ 0, 0, 0], [ 1, 1, 1]])
        self.k2 = torch.Tensor([[-1,  0,  1], [-1, 0, 1], [-1, 0, 1]])
        self.k3 = torch.Tensor([[ 1,  1,  1], [ 1, 1, 1], [ 1, 1, 1]])
        self.k4 = torch.Tensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [ 0,  0, 0, 0, 0], [ 1,  1, 1, 1, 1], [ 1,  1, 1, 1, 1]])
        self.k5 = torch.Tensor([[-1, -1,  0,  1,  1], [-1, -1,  0,  1,  1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1]])
        pass
    
    
    def forward(self, input_image):
        self.input_image = input_image
        #Conditions for Task 1,2,3 of Part A
        if self.mode == 'known':
            if self.o_channel == 1:
                self.taskno = 1
            elif self.o_channel == 2:
                self.taskno = 2
            elif self.o_channel == 3:
                self.taskno = 3
        #Conditions for Part B and C
        ###Either Part B will run or Part C, both cannot run together, comment out corresponing parts in main.py
        elif self.mode == 'rand':
            self.taskno = 4
            self.k_rand = torch.randn(self.kernel_size, self.kernel_size)   #kernel to be used for part b and part c
            #print(self.k_rand)
                                        
        [channel, img_height, img_width] = self.input_image.size()    #Find out dimensions of input image tensor  

        """TASK 1"""
        if self.taskno == 1:
            count_mul = 0
            count_add = 0
            outtensor = torch.zeros(img_height, img_width)     #Initialize the convolution output tensor for task 3
            kernel1 = torch.stack([self.k1 for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            for row in range(1,img_height-1):     # Loop over every pixel of the image
                for column in range(1,img_width-1):
                    out_tensor = torch.mul(kernel1,self.input_image[:,row-1:row+2,column-1:column+2])   #Perform element wise multiplication of the kernel with the roi of the image tensor
                    outtensor[row,column] = out_tensor.sum()    #Add all the elements and store in the output image tensor
                    count_mul +=1
                    count_add +=1
            outtensor = torch.unsqueeze(outtensor,0)            #Increase the output tensor in 1st dimension            
            #Calculate number of operations for all output channels
            #In each stage of torch.mul, each element of kernel is multiplied by image pixel, so no of multiplications is equal to no of element
            #Similarly, there are no of element - 1 no of additions
            #print(count_mul, count_add) 
            num_ops = count_mul * (torch.numel(kernel1)) + count_add * (torch.numel(kernel1) - 1)
        
        """TASK 2"""
        if self.taskno == 2:
            count_mul = 0
            count_add = 0
            outtensor = torch.zeros(self.o_channel, img_height, img_width)     #Initialize the convolution output tensor for task 2
            kernel1 = torch.stack([self.k4 for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            kernel2 = torch.stack([self.k5 for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            kernel = (kernel1,kernel2)
            for k in range(self.o_channel):        
                for row in range(2,img_height-2):     # Loop over every pixel of the image
                    for column in range(2,img_width-2):
                        out_tensor = torch.mul(kernel[k],self.input_image[:,row-2:row+3,column-2:column+3])   #Perform element wise multiplication of the kernel with the roi of the image  tensor
                        outtensor[k,row,column] = out_tensor.sum()    #Add all the elements and store in the output image tensor
                        count_mul +=1
                        count_add +=1
            #Calculate number of operations for all output channels
            #print(count_mul, count_add) 
            num_ops = count_mul * (torch.numel(kernel1)) + count_add * (torch.numel(kernel1) - 1)
            
        """TASK 3"""
        if self.taskno == 3:
            count_mul = 0
            count_add = 0
            outtensor = torch.zeros(self.o_channel, int(img_height/2), int(img_width/2))     #Initialize the convolution output tensor for task 3
            kernel1 = torch.stack([self.k1 for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            kernel2 = torch.stack([self.k2 for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            kernel3 = torch.stack([self.k3 for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            kernel = (kernel1,kernel2,kernel3)
            for k in range(self.o_channel):        
                for row in range(1,img_height-1,self.stride):     # Loop over every pixel of the image
                    for column in range(1,img_width-1,self.stride):
                        #print(kernel[k],self.input_image[:,row:row+5,column:column+5])
                        out_tensor = torch.mul(kernel[k],self.input_image[:,row-1:row+2,column-1:column+2])   #Perform element wise multiplication of the kernel with the roi of the image  tensor
                        outtensor[k,int(row/self.stride)+1,int(column/self.stride)+1] = out_tensor.sum()    #Add all the elements and store in the output image tensor
                        count_mul +=1
                        count_add +=1
            #Calculate number of operations for all output channels
            #print(count_mul, count_add) 
            num_ops = count_mul * (torch.numel(kernel1)) + count_add * (torch.numel(kernel1) - 1)                                   
        pass

        """PART B AND PART C"""
        #Same code for Part B and Part C
        if self.taskno == 4:
            count_mul = 0
            count_add = 0
            outtensor = torch.zeros(self.o_channel, img_height, img_width)     #Initialize the convolution output tensor for part B and C
            kernel1 = torch.stack([self.k_rand for i in range(self.in_channel)])   #Make a 3D kernel to by replicating the 2D kernel in the 3rd dimension
            #We will use same kernel multiple time, as many times as o_channel in part B and 2 times per iteratio in part C
            #print(kernel1.size())            
            roi_offset = int(self.kernel_size/2)
            #print('k size = %r, offset = %r'%(self.kernel_size,roi_offset))
            for k in range(self.o_channel):        
                for row in range(roi_offset,img_height-roi_offset):     # Loop over every pixel of the image
                    for column in range(roi_offset,img_width-roi_offset):
                        out_tensor = torch.mul(kernel1,self.input_image[:,row-roi_offset:row+roi_offset+1,column-roi_offset:column+roi_offset+1])   #Perform element wise multiplication of the kernel with the roi of the image  tensor
                        outtensor[k,row,column] = out_tensor.sum()    #Add all the elements and store in the output image tensor
                        count_mul +=1
                        count_add +=1
            #Calculate number of operations for all output channels
            #print(count_mul, count_add) 
            num_ops = count_mul * (torch.numel(kernel1)) + count_add * (torch.numel(kernel1) - 1)   #count_mul and count_add already considers all out channels, so multiply with elem(k1)
            
     
        return(num_ops,outtensor)   #return from forward function     
        
    #Function to normalize output and save separate image for each o_channel
    def normalize(self,img_no,output_img_array):
        self.output_img_array = output_img_array
        self.img_no = img_no
        #Normalize output image and Save each output channel as a separate grayscale image
        for i in range(self.o_channel):
            output_img_norm=(((self.output_img_array[i,:,:] - self.output_img_array[i,:,:].min()) / self.output_img_array[i,:,:].ptp()) * 255.0).astype(np.uint8)
            output_img_gray = Image.fromarray(output_img_norm)
            image_name = '/home/soumendu/Documents/Python /BME 595/Assignments/out_%s_task_%s_%s.jpg'%(self.img_no,self.taskno,i)
            output_img_gray.save(image_name)
        
        
