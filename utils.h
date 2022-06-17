#ifndef UTILS_H
#define UTILS_H

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

#endif
