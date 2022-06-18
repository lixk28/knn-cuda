#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CHECK macro from Grossman and McKercher, "Professional CUDA C Programming"
#define CHECK(call)                                         \
{                                                           \
    const cudaError_t error = call;                         \
    if (error != cudaSuccess) {                             \
        printf("Error: %s:%d, ", __FILE__, __LINE__);       \
        printf("code:%d, reason: %s \n",                    \
                error, cudaGetErrorString(error));          \
        exit(1);                                            \
    }                                                       \
}

#define divup(a, b) ((a) % (b) == 0 ? ((a) / (b)) : ((a) / (b) + 1))

#define min(a, b) ((a) < (b) ? (a) : (b))

/**
 *  @brief randomly generate test data
 *
 *  @x: query points
 *  @y: reference points
 *  @m: number of query points
 *  @n: number of reference points
 *  @d: point dimension
 *
 */
inline void generate_data(
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
    }

    for (int i = 0; i < n * d; i++) {
        y[i] = -500 + (1000.0f * rand()  / RAND_MAX );
    }

}

/**
 * @brief
 *
 * @param filename
 * @param m
 * @param k
 * @param indices
 * @param distance
 */
inline void output(
    const char *filename,
    int m,
    int k,
    int *indices,
    float *distance
)
{
    FILE* file;
    file = fopen(filename, "w");
    if(file != NULL) {
        for (int p = 0; p < m; p++) {
            for (int i = 0; i < k; i++) {
                fprintf(file, "%d %.5f", indices[p * k + i], distance[p * k + i]);
            }
        }
    }
    fclose(file);
}

/**
 * @brief check the error
 *
 * @param idx1
 * @param dist1
 * @param idx2
 * @param dist2
 * @param m
 * @param k
 */
inline void error_check(int *idx1, float *dist1, int *idx2, float *dist2, int m, int k)
{
    float threshold = 0.5f;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            if (idx1[i * k + j] != idx2[i * k + j]) {
                printf("error: the %d th near neighbour of the %d th point is different : ", j, i);
                printf("cuda : %d, omp : %d\n", idx1[i * k + j], idx2[i * k + j]);
                continue;
            }
            if (fabs(dist1[i * k + j] - dist2[i * k + j]) >= threshold) {
                printf("error: the dist from the %d th point to its %d th near neighbour is over threshold : ", i, j);
                printf("cuda : %f, omp : %f\n", dist1[i * k + j], dist2[i * k + j]);
                continue;
            }
        }
    }
}

/**
 * @brief gets the local time in nanoseconds
 */
inline long get_time()
{
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long)ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

#endif
