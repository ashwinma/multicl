nvcc -O3 -Xopencc="-LIST:source=on -O3" -arch=sm_21 -o gpuThrputBenchmark gpuThrputBenchmark.cu


nvcc -O0 -Xopencc="-LIST:source=on -O0" -arch=sm_21 -o gpuThrputBenchmark gpuThrputBenchmark.cu


nvcc -O0 -Xopencc="-LIST:source=on -O0" -arch=sm_20 -o gpuThrputBenchmark gpuThrputBenchmark.cu
