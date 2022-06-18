#ifndef KNN_H
#define KNN_H

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
    int *knn_idx,
    float *knn_dist
);

/**
 * @brief K-nearest neighbor implement with OpenMP
 *
 * params:
 *  @query:     query points
 *  @reference: reference points
 *  @m:         number of query points
 *  @n:         number of reference points
 *  @d:         point dimension
 *  @k:         number of nearest neighbors
 *  @indices:   index matrix
 *  @distance:  distance matrix
 */
void knn_omp(
    float *query,
    float *reference,
    int m,
    int n,
    int d,
    int k,
    int *indices,
    float *distance
);

#endif
