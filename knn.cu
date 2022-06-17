#include <stdio.h>
#include <math.h>
#include <cublas.h>
#include "knn.h"
#include "utils.h"

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


void knn_cuda(
    float *x,       // query points
    float *y,       // reference points
    int m,          // #query points
    int n,          // #reference points
    int d,          // point dimension
    int k,          // #neighbors
    int p,          // distance metric
    int *knn_idx,     // indices of k-nearest neighbors
    float *knn_dist   // distance of k-nearest neighbors
)
{
    // TODO: implement me
}
