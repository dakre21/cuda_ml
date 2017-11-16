#!/bin/bash 
#BSUB -n 1
#BSUB -o ml.out 
#BSUB -e ml.err 
#BSUB -q "windfall" 
#BSUB -J histogram
#BSUB -R gpu 

export FILE_LOCATION="pictures_location=./images"
#export SERIAL_FLAG="--serial" 

module load cuda/5.5.22
cd /extra/dakre/cuda_ml
time cuda_ml $FILE_LOCATION $SERIAL_FLAG 
###end of script 
