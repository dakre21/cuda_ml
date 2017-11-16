'''
Author: David Akre
Date: 11/15/17
Description: Main machine learning class to handle data prediction
'''

import sys
import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from scipy import ndimage
from pycuda.compiler import SourceModule

class Engine:

    def __init__(self, pic_loc, serial):
        self.pl = pic_loc
        self.ser = serial

    def _start_serial(self):
        """ 
        Function that will start serial execution
        """
        pass

    def _start_parallel(self):
        """
        Fuction that will start parallel execution
        """
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

    def start(self):
        """
        Function starts the execution of the main machine learning
        code-- either serially or in parallel
        """
        # TODO: Iterate through directory for file images

        if self.ser == True:
            self._start_serial()
        else:
            self._start_parallel()


