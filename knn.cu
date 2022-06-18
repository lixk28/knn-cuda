#include <stdio.h>
#include <math.h>
#include <cublas_v2.h>
#include "knn.h"
#include "utils.h"

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
    float *x_d,     // query points (stored in device memory)
    float *y_d,     // reference points (stored in device memory)
    int m,          // #query points (#rows of distance matrix)
    int n,          // #reference points (#cols of distance matrix)
    int d,          // point dimension
    float *dist_d   // distance matrix
)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float res = 0.0f;
        for (int i = 0; i < d; i++)
            res += (x_d[row * d + i] - y_d[col * d + i]) * (x_d[row * d + i] - y_d[col * d + i]);
        dist_d[row * n + col] = sqrtf(res);
#ifdef DEBUG
        printf("I'm thread (%d, %d), my result is %f\n", row, col, sqrtf(res));
#endif
    }
}

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
    float *dist_d,    // distance matrix of size m x n
    int m,            // #rows of distance matrix
    int n,            // #cols of distance matrix
    int k,            // #neighbors
    int *knn_idx_d,   // indices of k-nearest neighbors (stored in device memory)
    float *knn_dist_d // distance of k-nearest neighbors (stored in device memory)
)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // each thread sort one row (corresponding to one query point)
    if (tid < m) {
        float *my_dist_d     = dist_d     + tid * n;
        int   *my_knn_idx_d  = knn_idx_d  + tid * k;
        float *my_knn_dist_d = knn_dist_d + tid * k;

        my_knn_idx_d[0]  = 0;
        my_knn_dist_d[0] = my_dist_d[0];

        for (int i = 1; i < n; i++) {
            int   curr_idx  = i;
            float curr_dist = my_dist_d[i];

            // if the top k elements has been sorted (which means that i >= k)
            // and current value is bigger than the k-th value
            // then we do not have to sort this value
            if (i >= k && curr_dist >= my_dist_d[k - 1])
                continue;

            // shift values bigger than current value to the right
            // we only have to shift the top k elements
            int j = min(i - 1, k - 2);
            while (j >= 0 && curr_dist < my_dist_d[j]) {
                my_dist_d[j + 1]     = my_dist_d[j];
                my_knn_idx_d[j + 1]  = my_knn_idx_d[j];
                my_knn_dist_d[j + 1] = my_knn_dist_d[j];
                j--;
            }

            // write current index and value
            my_dist_d[j + 1]     = curr_dist;
            my_knn_idx_d[j + 1]  = curr_idx;
            my_knn_dist_d[j + 1] = curr_dist;
        }
    }
}

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
    float *x,         // query points
    float *y,         // reference points
    int m,            // #query points
    int n,            // #reference points
    int d,            // point dimension
    int k,            // #neighbors
    int p,            // distance metric
    int *knn_idx,     // indices of k-nearest neighbors
    float *knn_dist   // distance of k-nearest neighbors
)
{
    float *x_d        = NULL;
    float *y_d        = NULL;
    float *dist_d     = NULL;
    int   *knn_idx_d  = NULL;
    float *knn_dist_d = NULL;

    // allocate memory on device
    CHECK(cudaMalloc((void **)&x_d,        sizeof(float) * m * d));
    CHECK(cudaMalloc((void **)&y_d,        sizeof(float) * n * d));
    CHECK(cudaMalloc((void **)&dist_d,     sizeof(float) * m * n));
    CHECK(cudaMalloc((void **)&knn_idx_d,  sizeof(int)   * m * k));
    CHECK(cudaMalloc((void **)&knn_dist_d, sizeof(float) * m * k));

    // copy points from host to device
    CHECK(cudaMemcpy(x_d, x, sizeof(float) * m * d, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y_d, y, sizeof(float) * n * d, cudaMemcpyHostToDevice));

    // compute the distance matrix of query points and reference points
    dim3 grid(divup(n, 32), divup(m, 32));
    dim3 block(32, 32);
#ifdef DEBUG
    printf("grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
#endif
    compute_distance<<<grid, block>>>(x_d, y_d, m, n, d, dist_d);

#ifdef DEBUG
    float *dist_h = (float *)malloc(sizeof(float) * m * n);
    cudaMemcpy(dist_h, dist_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    printf("distance matrix:\n");
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            printf("%f%c", dist_h[i * n + j], j == n - 1 ? '\n' : '\t');
    free(dist_h);
#endif

    // sort the distance matrix to get k smallest
    k_selection_sort<<<divup(m, 256), 256>>>(dist_d, m, n, k, knn_idx_d, knn_dist_d);

#ifdef DEBUG
    float *dist_sorted_h = (float *)malloc(sizeof(float) * m * n);
    cudaMemcpy(dist_sorted_h, dist_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    printf("distance matrix after sorting (k smallest each row):\n");
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            printf("%f%c", dist_sorted_h[i * n + j], j == k - 1 ? '\n' : '\t');
    free(dist_sorted_h);
#endif

    // copy the result from device to host
    CHECK(cudaMemcpy(knn_idx,  knn_idx_d,  sizeof(int)   * m * k, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(knn_dist, knn_dist_d, sizeof(float) * m * k, cudaMemcpyDeviceToHost));

    // free memory allocated on device
    CHECK(cudaFree(x_d));
    CHECK(cudaFree(y_d));
    CHECK(cudaFree(dist_d));
    CHECK(cudaFree(knn_idx_d));
    CHECK(cudaFree(knn_dist_d));
}

/**
 * description:
 *  compute the squared norm vector given a set of points
 *  vec_i = ||pt_i||^2
 *
 * params:
 *  @pt: a set of points of size len x d
 *  @len: the number of points
 *  @d: the dimension of points
 *
 * returns:
 *  @vec: the squared norm vector
 *
 */
__global__ void compute_squared_norm_vector(
    float *pt,
    int len,
    int d,
    float *vec
)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < len) {
        float res = 0.0f;
        for (int i = 0; i < d; i++)
            res += pt[tid * d + i] * pt[tid * d + i];
        vec[tid] = res;
    }
}

/**
 * description:
 *  add vector to matrix
 *  given matrix `mat` of size m x n
 *  vector `x` of size m and vector `y` of size n
 *  mat[i][j] += (x[i] + y[j])
 *
 * params:
 *  @mat: the matrix of size m x n
 *  @vec_x_d: vector x (stored in device memory)
 *  @vec_y_d: vector y (stored in device memory)
 *  @m: the length of vec_x_d
 *  @n: the length of vec_y_d
 *
 * returns:
 *  @mat: the matrix of size m x n (has been added)
 *
 */
__global__ void add_vector_to_matrix(
    float *mat,
    float *vec_x_d,
    float *vec_y_d,
    int m,
    int n
)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float res = vec_x_d[row] + vec_y_d[col];
        mat[row * n + col] += res;
    }
}

/**
 * description:
 *  compute the squared root for each element of vector
 *
 * params:
 *  @vec: vector of size len
 *  @len: the length of vector
 *
 * returns:
 *  @vec: vector of size len (has been squared rooted)
 *
 */
__global__ void compute_squared_root(
    float *vec,
    int len
)
{
   int tid = blockDim.x * blockIdx.x + threadIdx.x;

   if (tid < len) {
       vec[tid] = sqrtf(vec[tid]);
   }
}

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
    float *x,         // query points
    float *y,         // reference points
    int m,            // #query points
    int n,            // #reference points
    int d,            // point dimension
    int k,            // #neighbors
    int p,            // distance metric
    int *knn_idx,     // indices of k-nearest neighbors
    float *knn_dist   // distance of k-nearest neighbors
)
{
    float *x_d        = NULL;
    float *y_d        = NULL;
    float *vec_x_d    = NULL;
    float *vec_y_d    = NULL;
    float *dist_d     = NULL;
    int   *knn_idx_d  = NULL;
    float *knn_dist_d = NULL;

    // allocate memory on device
    CHECK(cudaMalloc((void **)&x_d,        sizeof(float) * m * d));
    CHECK(cudaMalloc((void **)&y_d,        sizeof(float) * n * d));
    CHECK(cudaMalloc((void **)&vec_x_d,    sizeof(float) * m));
    CHECK(cudaMalloc((void **)&vec_y_d,    sizeof(float) * n));
    CHECK(cudaMalloc((void **)&dist_d,     sizeof(float) * m * n));
    CHECK(cudaMalloc((void **)&knn_idx_d,  sizeof(int)   * m * k));
    CHECK(cudaMalloc((void **)&knn_dist_d, sizeof(float) * m * k));

    // copy points from host to device
    CHECK(cudaMemcpy(x_d, x, sizeof(float) * m * d, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y_d, y, sizeof(float) * n * d, cudaMemcpyHostToDevice));

    // compute squared norm vector vec_x_d of size 1 x m
    // vec_x_d[i] is ||x_i||^2
    compute_squared_norm_vector<<<divup(m, 256), 256>>>(x_d, m, d, vec_x_d);

    // compute squared norm vector vec_y_d of size 1 x n
    // vec_y_d[i] is ||y_i||^2
    compute_squared_norm_vector<<<divup(n, 256), 256>>>(y_d, n, d, vec_y_d);

#ifdef DEBUG
    float *vec_x_h = (float *)malloc(sizeof(float) * m);
    float *vec_y_h = (float *)malloc(sizeof(float) * n);
    CHECK(cudaMemcpy(vec_x_h, vec_x_d, sizeof(float) * m, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(vec_y_h, vec_y_d, sizeof(float) * n, cudaMemcpyDeviceToHost));
    printf("squared norm vector of x:\n");
    for (int i = 0; i < m; i++)
        printf("%f%c", vec_x_h[i], i == m - 1 ? '\n' : '\t');
    printf("squared norm vector of y:\n");
    for (int i = 0; i < n; i++)
        printf("%f%c", vec_y_h[i], i == n - 1 ? '\n' : '\t');
    free(vec_x_h);
    free(vec_y_h);
#endif

    // use cublas sgemm to compute
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = -2.0f;
    float beta  =  0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d, &alpha, y_d, d, x_d, d, &beta, dist_d, n);

    cublasDestroy(handle);

    // add squared norm vector to get the squared distance matrix
    dim3 grid(divup(n, 32), divup(m, 32));
    dim3 block(32, 32);
    add_vector_to_matrix<<<grid, block>>>(dist_d, vec_x_d, vec_y_d, m, n);

#ifdef DEBUG
    float *dist_h = (float *)malloc(sizeof(float) * m * n);
    cudaMemcpy(dist_h, dist_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    printf("distance matrix (squared):\n");
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            printf("%f%c", dist_h[i * n + j], j == n - 1 ? '\n' : '\t');
    free(dist_h);
#endif

    // sort the distance matrix to get k smallest
    k_selection_sort<<<divup(m, 256), 256>>>(dist_d, m, n, k, knn_idx_d, knn_dist_d);

    // compute square root of knn_dist_d to get the real distance
    compute_squared_root<<<divup(m * k, 256), 256>>>(knn_dist_d, m * k);

    // copy the result from device to host
    CHECK(cudaMemcpy(knn_idx,  knn_idx_d,  sizeof(int)   * m * k, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(knn_dist, knn_dist_d, sizeof(float) * m * k, cudaMemcpyDeviceToHost));

    // free memory allocated on device
    CHECK(cudaFree(x_d));
    CHECK(cudaFree(y_d));
    CHECK(cudaFree(vec_x_d));
    CHECK(cudaFree(vec_y_d));
    CHECK(cudaFree(dist_d));
    CHECK(cudaFree(knn_idx_d));
    CHECK(cudaFree(knn_dist_d));
}

