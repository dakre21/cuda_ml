'''
Author: David Akre
Date: 11/15/17
Description: Main machine learning class to handle data prediction
'''

import numpy
import glob
import os
import csv
import sys
import pycuda.driver as cuda
import pycuda.autoinit
from time import time
from pycuda.compiler import SourceModule

class Engine:

    def __init__(self, pic_loc, serial):
        self.pl = pic_loc
        self.ser = serial
        self.mid_point = 128
        self.csv_dict = {}
        self.assessment = ""
        self.csv_file = "cuda_ml_results.csv"
        self.human_file = "human_assessment.txt"
  

    def _start_serial(self, images, hbuf):
        """ 
        Function that will start serial execution
        """

        for img in images:
            # Fwd declarations for calculations
            dark_count = 0
            light_count = 0
            exec_time = 0
            data = []
            tf = False
            img_bn = os.path.basename(img)

            # Read image
            with open(img) as i:
                img_buf = i.readlines()[0].split()

            # Determine light vs dark in image using img classification algo
            exec_time = time()

            for element in img_buf:
                if int(element) <= self.mid_point:
                    dark_count += 1
                else:
                    light_count += 1

            for res in hbuf:
                if img_bn in res:
                    self.assessment = res.split(",")[1].split("\n")[0]
                    break

            if dark_count >= light_count:
                if self.assessment != "dark":
                    self.mid_point += 10
                else:
                    tf = True

                data.append("dark")
            else:
                if self.assessment != "light":
                    self.mid_point -= 10
                else:
                    tf = True

                data.append("light")

            # Calculate total execution time
            exec_time = time() - exec_time

            # Append data to dictionary
            data.append(exec_time)
            data.append(tf)
            self.csv_dict[os.path.basename(img)] = data
    
    def _start_parallel(self, images, hbuf):
        """
        Fuction that will start parallel execution
        """

        for img in images:
            # Read image 
            with open(img) as i:
                img_buf = i.readlines()

            # Create convolutions

            # Determine light vs dark in image using image classification algo


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

        # Create data structure to create results 
        for i in range(len(raw_files)):
            self.csv_dict[os.path.basename(raw_files[i])] = []

        if not raw_files:
            print "ERROR: Directory provided did not contain any images"
            print "INFO: Correct image file extensions must be in raw format converted from pillow_utility.py"
            print "INFO: Reason is the UITS system will not support the pillow library due to incompatible dependencies... "\
                    "so this must be ran separately on a personal environment"
            return 

        print "Found appropriate image files within provided directory"
        print "Beginning image recognition algorithm on all of the images found within the directory..."

        with open(self.human_file, 'r') as f:
            hbuf = f.readlines()

        # Begin either serial or parallel image processing with the list of images
        if self.ser == True:
            self._start_serial(raw_files, hbuf)
        else:
            self._start_parallel(raw_files, hbuf)

        print "Finishing CUDA ML Application Execution"
        print "Results captured in " + self.csv_file

        # Create csv file
        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            for key, value in self.csv_dict.items():
                writer.writerow([key, value])



