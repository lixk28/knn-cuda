#ifndef KNN_H
#define KNN_H

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * description:
 *  cuda kernel for computing the distance matrix
 *
 * params:
 *  @x_d: query points stored in device memory
 *  @y_d: reference points stored in device memory
 *  @m: number of query points
 *  @n: number of reference points
 *  @d: dimension of points
 *
 * returns:
 *  @dist_d: distance matrix of size m x n from query points to reference points
 *
 */
__global__ void compute_distance(
    float *x_d,
    float *y_d,
    int m,
    int n,
    int d,
    float *dist_d
);

/**
 * description:
 *  cuda kernel for sorting the k smallest points
 *  for each row of distance matrix, find the top k smallest values
 *
 * params:
 *  @dist_d: distance matrix of size m x n
 *  @m: number of query points (#rows of dist_d)
 *  @n: number of reference points (#cols of dist_d)
 *  @k: number of nearest neighbors
 *
 * returns:
 *  @knn_idx_d: indices of k-nearest neighbors of size m x k
 *  @knn_dist_d: distance of k-nearest neighbors of size m x k
 *
 */
__global__ void k_selection_sort(
    float *dist_d,
    int m,
    int n,
    int k,
    int *knn_idx_d,
    float *knn_dist_d
);

/**
 * description:
 *  k-nearest neighbor search using cuda
 *
 * params:
 *  @x: query points
 *  @y: reference points
 *  @m: number of query points
 *  @n: number of reference points
 *  @d: point dimension
 *  @k: number of nearest neighbors
 *  @p: power parameter for the minkowski metric
 *      - when p = 1, it's equivalent to manhattan distance
 *      - when p = 2, it's equivalent to euclidean distance
 *
 * returns:
 *  @knn_idx: indices of the nearest points
 *  @knn_dist: distance from the nearest points
 *
 */
void knn_cuda(
    float *x,
    float *y,
    int m,
    int n,
    int d,
    int k,
    int p,
    int *knn_idx,
    float *knn_dist
);

/**
 * description:
 *  k-nearest neighbor search using cublas
 *  the computation of distance matrix can be converted to matrix operations
 *  which can accelerated by cublas
 *
 * params:
 *  @x: query points
 *  @y: reference points
 *  @m: number of query points
 *  @n: number of reference points
 *  @d: point dimension
 *  @k: number of nearest neighbors
 *  @p: power parameter for the minkowski metric
 *      - when p = 1, it's equivalent to manhattan distance
 *      - when p = 2, it's equivalent to euclidean distance
 *
 * returns:
 *  @knn_idx: indices of the nearest points
 *  @knn_dist: distance from the nearest points
 *
 */
void knn_cublas(
    float *x,
    float *y,
    int m,
    int n,
    int d,
    int k,
    int p,
    int *knn_idx,
    float *knn_dist
);

#endif
