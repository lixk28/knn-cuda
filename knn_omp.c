#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define DEBUG

/**
 * @brief compute the distance between query points and reference points
 *
 * @param query         the float array of query points
 * @param reference     the float array of reference points
 * @param distance      float array to store the distances
 * @param n             the num of query points
 * @param m             the num of reference points
 * @param d             dimension of points
 */
void compute_distance(
    float* query,
    float* reference,
    float* distance,
    int n,
    int m,
    int d
)
{
    #pragma parallel for
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            for(int k = 0; k < d; k++){
                #pragma omp atomic
                distance[i * m + j] += (query[i * d + k] - reference[j * d + k]) * (query[i * d + k] - reference[j * d + k]);
            }
        }
    }
}

/**
 * @brief odd even sort with openmp parallel
 *
 * @param row
 * @param n     the num of elements
 */
void odd_even_sort(float* row, int* indices, int n){

    int flag = 1;

    omp_set_num_threads(n / 2);
    while (flag)
    {
        flag = 0;

        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2)
        {
            if (row[i] > row[i + 1])
            {
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
        for (int i = 1; i < n - 1; i += 2)
        {
            if (row[i] > row[i + 1])
            {
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

// // getTime gets the local time in nanoseconds.
// long get_time() {
//     struct timespec ts;
//     timespec_get(&ts, TIME_UTC);
//     return (long)ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
// }

/**
 * @brief
 *
 * @param x
 * @param y
 * @param m
 * @param n
 * @param d
 * @return void*
 */
void* generate_data(
    float *x,     // query points
    float *y,     // reference points
    int m,        // #query points
    int n,        // #reference points
    int d         // point dimension
)
{
    srand(time(NULL));
    for (int i = 0; i < m * d; i++) {
        x[i] = -500 + (1000.0f * rand()  / RAND_MAX );
        printf("%f ", x[i]);
    }
    printf("\n");
    for (int i = 0; i < n * d; i++) {
        y[i] = -500 + (1000.0f * rand()  / RAND_MAX );
        printf("%f ", y[i]);
    }
    printf("\n");

}

int main(){
    const int d = 3;    //dimension
    const int m = 10;    //the num of reference points
    const int n = 3;    //the num of query points
    const int k = 10;    //the num of neighbors

    float* query = (float*)malloc(sizeof(float) * n * d);
    float* reference = (float*)malloc(sizeof(float) * m * d);
    float* distance = (float*)malloc(sizeof(float) * n * m);
    int* indices = (int*)malloc(sizeof(int) * m * n);
    generate_data(query, reference, n, m, d);

    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            indices[i * m + j] = j;
        }
    }

    //数据生成

    compute_distance(query, reference, distance, n, m, d);

    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        odd_even_sort(distance + i * m, indices + i * m, m);
    }

#ifdef DEBUG
    for (int p = 0; p < n; p++) {
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