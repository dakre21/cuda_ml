# CUDA Machine Learning Demo
## Pre-requisites
- System that has an NVIDIA GPGPU and CUDA installed
- Load cuda (e.g. module load cuda/5.5.22 after sourcing venv environment script... note the newest 8.0.61 is not working)

## Setup instructions
1. Create virtualenv
- ```virtualenv venv --system-site-packages```
2. Source venv activate script
- ```source venv/bin/activate```
3. Install dependencies and project
- ```python setup.py install```

## Run instructions
1. Simply run either:
a. bsub < submit.sh (on UITS system)
- Note: Modify submit.sh with the respective image directory and serial flag (e.g. if its present or not... currently disabled to
run in parallel)
b. cuda_ml <path_to_images> 

## Misc
For help on running simply run cuda_ml --help to display options
