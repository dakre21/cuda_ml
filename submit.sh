#!/bin/bash 
#BSUB -n 1
#BSUB -o ml.out 
#BSUB -e ml.err 
#BSUB -q "windfall" 
#BSUB -J histogram
#BSUB -R gpu 

module load cuda/5.5.22
cd /extra/dakre/cuda_ml
time cuda_ml 
###end of script 
