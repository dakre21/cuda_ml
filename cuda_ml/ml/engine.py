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
        self.csv_dict = {}
        self.assessment = ""
        self.csv_serial_file = "cuda_ml_serial_results.csv"
        self.csv_parallel_file = "cuda_ml_parallel_results.csv"
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
            mid_point = 128
            data = []
            tf = False
            img_bn = os.path.basename(img)

            # Read image
            with open(img) as i:
                img_buf = i.readlines()[0].split()

            # Determine light vs dark in image using img classification algo
            exec_time = time()

            for element in img_buf:
                if int(element) <= mid_point:
                    dark_count += 1
                else:
                    light_count += 1

            for res in hbuf:
                if img_bn in res:
                    self.assessment = res.split(",")[1].split("\n")[0]
                    break

            if dark_count >= light_count:
                if self.assessment != "dark":
                    mid_point += 10
                else:
                    tf = True

                data.append("dark")
            else:
                if self.assessment != "light":
                    mid_point -= 10
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
            # Fwd declarations for calculations
            exec_time = 0
            light_count = 0
            dark_count = 0
            mid_point = [128]
            data = []
            tf = False
            img_bn = os.path.basename(img)

            # Read image
            with open(img) as i:
                img_buf = i.readlines()[0].split()

            size = [len(img_buf)]

            # Convert to single precision
            img_buf = numpy.array(img_buf).astype(numpy.uint32)
            mid_point = numpy.array(mid_point).astype(numpy.uint32)
            size = numpy.array(size).astype(numpy.uint32)

            # Allocate memory for gpu
            img_gpu = cuda.mem_alloc(img_buf.nbytes)
            mid_point_gpu = cuda.mem_alloc(mid_point.nbytes)
            size_gpu = cuda.mem_alloc(size.nbytes)

            # Transfer data onto gpu
            cuda.memcpy_htod(img_gpu, img_buf)
            cuda.memcpy_htod(mid_point_gpu, mid_point)
            cuda.memcpy_htod(size_gpu, size)

            # Declare CUDA kernel function
            mod = SourceModule("""
            __global__ void detector(unsigned int* img_buf, unsigned int* mid_point, unsigned int* size) {
                for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size[0]; i += blockDim.x * gridDim.x) {
                    if (img_buf[i] <= mid_point[0]) {
                        img_buf[i] *= 0;
                    }
                }
            } 
            """)

            # Determine light vs dark in image using img classification algo
            start = cuda.Event()
            end = cuda.Event()

            # Invoke kernel function
            func = mod.get_function("detector")
            start.record() 
            func(img_gpu, mid_point_gpu, size_gpu, block=(512, 2, 1), grid=(512, 2, 1))
            end.record() 
            end.synchronize()
            exec_time = start.time_till(end)*1e-3

            # Fetch data from kernel
            new_img_buf = numpy.empty_like(img_buf)
            cuda.memcpy_dtoh(new_img_buf, img_gpu)

            for res in hbuf:
                if img_bn in res:
                    self.assessment = res.split(",")[1].split("\n")[0]
                    break

            for element in new_img_buf:
                if element == 0:
                    dark_count += 1
                else:
                    light_count += 1

            if dark_count >= light_count:
                if self.assessment != "dark":
                    mid_point += 10
                else:
                    tf = True

                data.append("dark")
            else:
                if self.assessment != "light":
                    mid_point -= 10
                else:
                    tf = True

                data.append("light")

            # Append data to dictionary
            data.append(exec_time)
            data.append(tf)
            self.csv_dict[os.path.basename(img)] = data

  
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
            csv_file = self.csv_serial_file
        else:
            self._start_parallel(raw_files, hbuf)
            csv_file = self.csv_parallel_file

        print "Finishing CUDA ML Application Execution"
        print "Results captured in " + csv_file

        # Create csv file
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            for key, value in self.csv_dict.items():
                writer.writerow([key, value])



