# GPU-accelerated k-nearest neighbors search

This repository contains our final project for DCS316: Multicore Programming lectured by [Prof. Tao (Jun Tao)](http://www.juntao.pro) at [Sun Yat-sen University](https://www.sysu.edu.cn/sysuen).

We reproduced and further optimized kNN algorithms proposed in the following papers:

- [Fast k Nearest Neighbor Search using GPU, Vincent Garcia](https://arxiv.org/pdf/0804.1448)
- [k-Nearest Neighbor Search: Fast GPU-Based Implementations and Application to High-Dimensional Feature Space, Vincent Garcia](https://vincentfpgarcia.github.io/data/Garcia_2010_ICIP.pdf)

There are 3 versions of kNN in our implementation:

1. kNN OpenMP
2. kNN CUDA
3. kNN CUBLAS (accelerated by `cublasSgemm` based on kNN CUDA)

CUBLAS version is way more faster than CUDA version for high dimensional data, and it's our best implementation for GPU-accelerated kNN.
