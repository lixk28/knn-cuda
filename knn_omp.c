#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

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
            for(int k = 0; k < d; k++) {
                distance[i * n + j] += (query[i * d + k] - reference[j * d + k]) * (query[i * d + k] - reference[j * d + k]);
            }
        }
    }
}

/**
 * @brief odd even sort with openmp parallel
 *
 * @param row   the row to be sorted
 * @param n     the num of elements
 */
static void odd_even_sort(float *row, int *indices, int n)
{
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

                int temp2 = indices[i];
                indices[i] = indices[i + 1];
                indices[i + 1] = temp2;
            }
        }

        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (row[i] > row[i + 1]) {
                flag = 1;
                float temp = row[i];
                row[i] = row[i + 1];
                row[i + 1] = temp;

                int temp2 = indices[i];
                indices[i] = indices[i + 1];
                indices[i + 1] = temp2;
            }
        }
    }
}

/**
 * @brief K-nearest neighbor implement with OpenMp
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
    float *query,       // query points
    float *reference,   // reference points
    int m,              // number of query points
    int n,              // number of reference points
    int d,              // point dimension
    int k,              // number of nearest neighbors
    int *indices,     // index matrix
    float *distance     // distance matrix
)
{
    generate_data(query, reference, m, n, d);
    #pragma omp parallel for
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            indices[i * n + j] = j;
        }
    }

    compute_distance(query, reference, distance, m, n, d);

    #pragma omp parallel for
    for(int i = 0; i < m; i++) {
        odd_even_sort(distance + i * n, indices + i * n, n);
    }

#ifdef DEBUG
    for (int p = 0; p < m; p++) {
        printf("%d-nearest neighbor of ", k);
        printf("(");
        for (int i = 0; i < d; i++) {
            if (i == d - 1)
                printf("%f)\n", query[p * d + i]);
            else
                printf("%f, ", query[p * d + i]);
        }
        for (int i = 0; i < k; i++) {
            printf("(");
            for (int j = 0; j < d; j++) {
                if (j == d - 1)
                    printf("%f): %f\n", reference[indices[p * k + i] * d + j], distance[p * k + i]);
                else
                    printf("%f, ", reference[indices[p * k + i] * d + j]);
            }
        }
        printf("\n\n");
    }
#endif

}
