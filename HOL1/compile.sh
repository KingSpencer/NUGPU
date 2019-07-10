#!/bin/bash
nvcc -arch=sm_35 -O3 VectorAdd.cu -o vAdd
