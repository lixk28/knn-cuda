#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
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

__global__ void warmup()
{
    // warmup kernel
    // do nothing
}


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
void generate_data(
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
void output(
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

#endif
