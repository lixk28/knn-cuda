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
