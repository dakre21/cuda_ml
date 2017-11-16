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
            # Read image
            with open(img) as i:
                img_buf = i.readlines()

            # Determine light vs dark in image

            # Performan machine learning regression based on result


    
    def _start_parallel(self, images):
        """
        Fuction that will start parallel execution
        """

        for img in images:
            # Read image 
            with open(img) as i:
                img_buf = i.readlines()

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
        raw = os.getcwd() + "/" + self.pl + "/*.raw"
        raw_files = glob.glob(raw)

        if not raw_files:
            print "ERROR: Directory provided did not contain any images"
            print "INFO: Correct image file extensions must be in raw format converted from pillow_utility.py"
            print "INFO: Reason is the UITS system will not support the pillow library due to incompatible dependencies... "\
                    "so this must be ran separately on a personal environment"
            return 

        # Begin either serial or parallel image processing with the list of images
        if self.ser == True:
            self._start_serial(raw_files)
        else:
            self._start_parallel(raw_files)


