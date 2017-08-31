#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

long double c_conv(int in_channel, long int o_channel, int kernel_size, int stride, float ***input_image);

int main(int argc, char** argv)
{
    int in_channel, kernel_size, stride;
    int i,j,k;
    long int o_channel;
    int rows, columns;
    float ***img_array;
    double computation_time[11];
    long double num_of_ops;
    clock_t start, end;

    //Image 0 size
    rows = 720;
    columns = 1280;
    
    //Image 1 size
    //rows = 1080;
    //columns = 1920;
    in_channel = 3;
    kernel_size = 3;
    stride = 1;
    
    //Allocate space for 3D image array
    img_array = (float***)malloc(in_channel*sizeof(float**));
    for (i = 0; i < in_channel; i++)
        img_array[i] = (float**)malloc(rows*sizeof(float*));
    
    for (i = 0; i < in_channel; i++) 
        for (j = 0; j < rows; j++) 
            img_array[i][j] = (float*)malloc(columns*sizeof(float));
       
    /*Create input test image*/
    for(i = 0; i < in_channel; i++)
        for(j = 0; j < rows; j++)
            for(k = 0; k < columns; k++){
                img_array[i][j][k] = rand()%255;
                //printf("\n%f",img_array[i][j][k]);
            }
                
    for(i = 0; i < 11; i++)
        computation_time[i] = 0; //Variable initialization for time taken
   
    //Initialize Conv2D
    printf("\n------------------Part D--------------------\n");
    for(i = 0; i <11; i++)      
    {
        o_channel = pow(2,i);
        start = clock();
        num_of_ops = c_conv(in_channel, o_channel, kernel_size, stride, img_array);   //Call c_conv function
        end = clock();
        computation_time[i] = (double)(end - start) / CLOCKS_PER_SEC;
        printf("i = %d, o_channel = %ld, number_of_operations = %Lf, computation_time = %lf \n", i, o_channel, num_of_ops, computation_time[i]);
    }
    return 0;

}


long double c_conv(int in_channel, long int o_channel, int kernel_size, int stride, float ***input_image)
{
    float ***kernel, ***out_img_array; //kernel and output array declaration 
    int i,j,k,a,b,c,x,y,z,roi_offset;
    int num_of_ops = 0;
    int count_mul = 0, count_add = 0;
    float mul_acc = 0;
    long double count = 0;
    
    //Image 0 size
    int rows = 720;
    int columns = 1280;
    
    //Image 1 size
    //int rows = 1080;
    //int columns = 1920;
    
    //Allocate space for 3D kernel
    kernel = (float***)malloc(kernel_size*sizeof(float**));
    for (i = 0; i < kernel_size; i++)
        kernel[i] = (float**)malloc(kernel_size*sizeof(float*));
    
    for (i = 0; i < kernel_size; i++) 
        for (j = 0; j < kernel_size; j++) 
            kernel[i][j] = (float*)malloc(kernel_size*sizeof(float));
    
    //Create 3D kernel
    for(i = 0; i < kernel_size; i++)
        for(j = 0; j < kernel_size; j++)
            for(k = 0; k < kernel_size; k++){
                kernel[i][j][k] = (float)(rand()%20-10.0)/10.0;
                //printf("\n%f",kernel[i][j][k]);
            }
          
                        
    //Allocate space for 3D output array
    out_img_array = (float***)malloc(o_channel*sizeof(float**));
    for (i = 0; i < o_channel; i++)
        out_img_array[i] = (float**)malloc(rows*sizeof(float*));
    
    for (i = 0; i < o_channel; i++) 
        for (j = 0; j < rows; j++) 
            out_img_array[i][j] = (float*)malloc(columns*sizeof(float));
       
    /*Initialize output image with all zeros*/
    for(i = 0; i < o_channel; i++)
        for(j = 0; j < rows; j++)
            for(k = 0; k < columns; k++)
                out_img_array[i][j][k] = 0;
                
    roi_offset = (int)(kernel_size/2);
    //printf("\nk size = %d, offset = %d",kernel_size,roi_offset);
    
    //Loop over every pixel of the image
    for(i = 0; i < o_channel; i++)
        for(j = roi_offset; j < rows-roi_offset; j++)
            for(k = roi_offset; k < columns-roi_offset; k++)
            {
                mul_acc = 0; //Initialize variable for each step as kernel is shifted through the image
                //Multiply and accumulate region of image of size 3x3x3 with kernel of 3x3x3
                for(x = 0, a = 0; x < kernel_size; x++, a++){
                    for(y = 0, b = j-roi_offset; y < kernel_size; y++, b++){
                        for(z = 0, c = k-roi_offset; z < kernel_size; z++, c++){
                            mul_acc += kernel[x][y][z] * input_image[a][b][c];
                            count += 2;
                            //printf("i%d, j%d, k%d, a%d, b%d, c%d, x%d, y%d, z%d, %f \n",i,j,k,a,b,c,x,y,z,mul_acc);
                        }//end for
                        //printf("\n");
                    }
                    //printf("\n\n");
                }//end mul and acc for
                out_img_array[i][j][k] = mul_acc;
             }

    return count; //num_of_ops;
}


