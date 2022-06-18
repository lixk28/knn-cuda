#include <stdio.h>
#include <stdlib.h>
#include "knn.h"
#include "utils.h"

#define N_NUM 8
#define D_NUM 5

int main(int argc, char *argv[])
{
    int k = 20;
    int ns[N_NUM];
    int ds[D_NUM];
    for (int i = 0; i < N_NUM; i++)
        ns[i] = 256 << i;
    for (int j = 0; j < D_NUM; j++)
        ds[j] = 1 << (j * 2);

    long cuda_time[N_NUM][D_NUM];
    long cublas_time[N_NUM][D_NUM];
    long omp_time[N_NUM][D_NUM];

    float *x         = NULL;
    float *y         = NULL;
    int   *cuda_idx  = NULL;
    float *cuda_dist = NULL;
    int   *omp_idx   = NULL;
    float *omp_dist  = NULL;

    for (int i = 0; i < N_NUM; i++) {
        for (int j = 0; j < D_NUM; j++) {
            int m = ns[i];
            int n = ns[i];
            int d = ds[j];

            x = (float *)malloc(sizeof(float) * m * d);
            y = (float *)malloc(sizeof(float) * n * d);

            // generate points randomly
            generate_data(x, y, m, n, d);

            // allocate memory for storing the results
            cuda_idx  = (int *)  malloc(sizeof(int)   * m * k);
            cuda_dist = (float *)malloc(sizeof(float) * m * k);
            omp_idx   = (int *)  malloc(sizeof(int)   * m * k);
            omp_dist  = (float *)malloc(sizeof(float) * m * k);

            long start_time, end_time;

            // knn cuda version
            start_time = get_time();
            knn_cuda(x, y, m, n, d, k, cuda_idx, cuda_dist);
            end_time = get_time();
            printf("with configuration m = n = %d, d = %d, k = %d\n", m, d, k);
            printf("  knn cuda version : %ld ms\n", end_time - start_time);
            cuda_time[i][j] = end_time - start_time;

            // knn cublas version
            start_time = get_time();
            knn_cublas(x, y, m, n, d, k, cuda_idx, cuda_dist);
            end_time = get_time();
            printf("  knn cublas version : %ld ms\n", end_time - start_time);
            cublas_time[i][j] = end_time - start_time;

            // knn openmp version
            start_time = get_time();
            knn_omp(x, y, m, n, d, k, omp_idx, omp_dist);
            end_time = get_time();
            printf("  knn omp version : %ld ms\n", end_time - start_time);
            omp_time[i][j] = end_time - start_time;

            error_check(cuda_idx, cuda_dist, omp_idx, omp_dist, m, k);

            free(x);
            free(y);
            free(cuda_idx);
            free(cuda_dist);
            free(omp_idx);
            free(omp_dist);
        }
    }

    FILE *fp;
    fp = fopen("time.txt", "w");
    for (int i = 0; i < N_NUM; i++) {
        for (int j = 0; j < D_NUM; j++)
            fprintf(fp, "%ld ", cuda_time[i][j]);
        fprintf(fp, "\n");
    }
    for (int i = 0; i < N_NUM; i++) {
        for (int j = 0; j < D_NUM; j++)
            fprintf(fp, "%ld ", cublas_time[i][j]);
        fprintf(fp, "\n");
    }
    for (int i = 0; i < N_NUM; i++) {
        for (int j = 0; j < D_NUM; j++)
            fprintf(fp, "%ld ", omp_time[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}
