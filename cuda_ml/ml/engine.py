'''
Author: David Akre
Date: 11/15/17
Description: Main machine learning class to handle data prediction
'''

import numpy
import glob
import os
import pycuda.driver as cuda
import pycuda.autoinit
from scipy import ndimage
from pycuda.compiler import SourceModule

class Engine:

    def __init__(self, pic_loc, serial):
        self.pl = pic_loc
        self.ser = serial

   
    def _start_serial(self, images):
        """ 
        Function that will start serial execution
        """

        for img in images:
            # Read image and convert to greyscale array
            bl_img = ndimage.imread(img, flatten=True)
            print bl_img

            # Determine light vs dark in image

            # Performan machine learning regression based on result


    
    def _start_parallel(self, images):
        """
        Fuction that will start parallel execution
        """

        for img in images:
            # Read image and convert to greyscale array
            bl_img = ndimage.imread(img, flatten=True)
            print bl_img

            # Determine light vs dark in image

            # Performan machine learning regression based on result


        '''
        # Create random matrix
        a = numpy.random.randn(4,4)

        # Convert to single precision
        a = a.astype(numpy.float32)

        # Allocate memory for gpu
        a_gpu = cuda.mem_alloc(a.nbytes)

        # Transfer data onto gpu
        cuda.memcpy_htod(a_gpu, a)

        # Declare CUDA kernel function
        mod = SourceModule("""
        __global__ void doublify(float *a)
        {
            int idx = threadIdx.x + threadIdx.y*4;
            a[idx] *= 2;
        } 
        """)

        # Invoke kernel function
        func = mod.get_function("doublify")
        func(a_gpu, block=(4,4,1))

        # Fetch data from kernel
        a_doubled = numpy.empty_like(a)
        cuda.memcpy_dtoh(a_doubled, a_gpu)
        print a_doubled
        print a
        '''

  
    def start(self):
        """
        Function starts the execution of the main machine learning
        code-- either serially or in parallel
        """

        # Verify directory provided contains images
        ppm = os.getcwd() + "/" + self.pl + "/*.ppm"
        png = os.getcwd() + "/" + self.pl + "/*.png"
        jpg = os.getcwd() + "/" + self.pl + "/*.jpg"
        jpeg = os.getcwd() + "/" + self.pl + "/*.jpeg"

        ppm_files = glob.glob(ppm)
        png_files = glob.glob(png)
        jpg_files = glob.glob(jpg)
        jpeg_files = glob.glob(jpeg)

        if not ppm_files and not png_files and not jpg_files and not jpeg_files:
            print "ERROR: Directory provided did not contain any images"
            print "INFO: Correct image file extensions must be .ppm, .png, .jpg, or .jpeg"
            return 

        images = ppm_files + png_files + jpg_files + jpeg_files

        # Begin either serial or parallel image processing with the list of images
        if self.ser == True:
            self._start_serial(images)
        else:
            self._start_parallel(images)


