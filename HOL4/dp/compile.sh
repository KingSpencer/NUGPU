nvcc -arch=sm_35 -O3 modified2.cu -o modified2 -Xcompiler "-fopenmp" -rdc=true
