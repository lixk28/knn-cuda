#include <stdio.h>
#include <stdlib.h>
#include "knn.h"

int main(int argc, char *argv[])
{

    float x[] = {
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f
    };

    float y[] = {
        1.0f, 1.0f,
        1.0f, 2.0f,
        0.5f, 0.5f,
        -1.0f, 0.0f,
        0.0f, -1.0f
    };

    int m = 3;
    int n = 5;
    int d = 2;
    int k = 2;

    int *knn_idx = (int *)malloc(sizeof(int) * m * k);
    float *knn_dist = (float *)malloc(sizeof(float) * m * k);

    knn_cuda(x, y, m, n, d, k, knn_idx, knn_dist);

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
    }
#endif

    free(knn_idx);
    free(knn_dist);

    return 0;
}
