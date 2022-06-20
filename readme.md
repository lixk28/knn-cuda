# GPU-accelerated nearest neighbor search

This repository contains our final project for multicore programming lectured by [Prof. Tao (Jun Tao)](http://www.juntao.org/) at [Sun Yat-sen University](https://www.sysu.edu.cn/).

We basically reproduced the results in this two papers:

- [Fast k Nearest Neighbor Search using GPU, Vincent Garcia](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.9386&rep=rep1&type=pdf)
- [k-Nearest Neighbor Search: Fast GPU-Based Implementations and Application to High-Dimensional Feature Space, Vincent Garcia](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=064592F4CB50DE20D130694460A83334?doi=10.1.1.172.2896&rep=rep1&type=pdf)

We implemented three versions for kNN (using exhaustive search):

- kNN OpenMP
- kNN CUDA
- kNN CUBLAS (use `cublasSgemm` to accelerate)

OpenMP version is for comparison, CUBLAS version is way more faster than CUDA version for high dimensional data, and it's our best implementation for this problem.
