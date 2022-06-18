#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "knn.h"

/**
 * @brief compute the distance between query points and reference points
 *
 * @param query         the float array of query points
 * @param reference     the float array of reference points
 * @param distance      float array to store the distances
 * @param m             the num of query points
 * @param n             the num of reference points
 * @param d             dimension of points
 */
static void compute_distance(
    float *query,
    float *reference,
    float *distance,
    int m,
    int n,
    int d
)
{
    #pragma parallel for
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            distance[i * n + j] = 0;
            for(int k = 0; k < d; k++) {
                distance[i * n + j] += (query[i * d + k] - reference[j * d + k]) * (query[i * d + k] - reference[j * d + k]);
            }
            distance[i * n + j] = sqrtf(distance[i * n + j]);
        }
    }
}


/**
 * @brief odd even sort with OpenMP parallel
 *
 * @param row       the row to be sorted
 * @param row_res   first k elements of the sorted row
 * @param n         the num of elements
 * @param k         first k element
 */
static void odd_even_sort(float *row, float *row_res, int *indices, int n, int k)
{
    int *indices_tmp = (int*)malloc(sizeof(int) * n);

    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        indices_tmp[i] = i;

    int flag = 1;

    omp_set_num_threads(n / 2);
    while (flag) {

        flag = 0;
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            if (row[i] > row[i + 1]) {
                flag = 1;
                float temp = row[i];
                row[i] = row[i + 1];
                row[i + 1] = temp;

                int temp2 = indices_tmp[i];
                indices_tmp[i] = indices_tmp[i + 1];
                indices_tmp[i + 1] = temp2;
            }
        }

        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (row[i] > row[i + 1]) {
                flag = 1;
                float temp = row[i];
                row[i] = row[i + 1];
                row[i + 1] = temp;

                int temp2 = indices_tmp[i];
                indices_tmp[i] = indices_tmp[i + 1];
                indices_tmp[i + 1] = temp2;
            }
        }
    }

    #pragma omp parallel for
    for(int j = 0; j < k; j++) {
        row_res[j] = row[j];
        indices[j] = indices_tmp[j];
    }

}

/**
 * @brief K-nearest neighbor implement with OpenMP
 *
 * params:
 *  @x: query points
 *  @y: reference points
 *  @m: number of query points
 *  @n: number of reference points
 *  @d: point dimension
 *  @k: number of nearest neighbors
 *  @knn_idx: index matrix
 *  @knn_dist: distance matrix
 *
 * returns:
 *  @knn_idx: indices of the nearest points
 *  @knn_dist: distance from the nearest points
 */
void knn_omp(
    float *x,
    float *y,
    int m,
    int n,
    int d,
    int k,
    int *knn_idx,
    float *knn_dist
)
{
    #pragma omp parallel for
    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            knn_idx[i * n + j] = j;

    float *distance = (float*)malloc(sizeof(float) * m * n);
    compute_distance(x, y, distance, m, n, d);

    #pragma omp parallel for
    for(int i = 0; i < m; i++)
        odd_even_sort(distance + i * n, knn_dist + i * k, knn_idx + i * k, n, k);

#ifdef DEBUG
    for (int p = 0; p < m; p++) {
        printf("%d-nearest neighbor of ", k);
        printf("(");
        for (int i = 0; i < d; i++) {
            if (i == d - 1)
                printf("%f)\n", x[p * d + i]);
            else
                printf("%f, ", x[p * d + i]);
        }
        for (int i = 0; i < k; i++) {
            printf("(");
            for (int j = 0; j < d; j++) {
                if (j == d - 1)
                    printf("%f): %f\n", y[knn_idx[p * k + i] * d + j], knn_dist[p * k + i]);
                else
                    printf("%f, ", y[knn_idx[p * k + i] * d + j]);
            }
        }
        printf("\n\n");
    }
#endif
}
