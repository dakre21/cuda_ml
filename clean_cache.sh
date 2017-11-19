#!/bin/bash

echo "Cache currently contains"
ls /home/u32/`whoami`/.cache/pycuda/compiler-cache-v1/

echo "Cleaning cuda cache..."
rm -rf /home/u32/`whoami`/.cache/pycuda/compiler-cache-v1/*

echo "Visually inspect cuda cache..."
ls /home/u32/`whoami`/.cache/pycuda/compiler-cache-v1/
echo "If nothing printed from ls then you're good to go!"
