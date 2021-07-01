#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "sample_cuda_kernel.h"

#define BLOCK 512

dim3 cuda_gridsize(int n)
{
    int k = (n-1) / BLOCK + 1;
    int x = k;
    int y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

__global__ void sample_forward_kernel(int N,  float const *features,float const *features_loc, int num_sample, int band, float const *spa_spe, float *adj_sample_spa, float *adj_sample_spe)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%num_sample;
    i = i/num_sample;
    int in_h = i%num_sample;
    if(in_w<in_h) return;
    float mse = 0;
    for(int i=0;i<2;i++)
    {
        int ind1 = in_w+num_sample*i;
        int ind2 = in_h+num_sample*i;
        mse = mse +pow(features_loc[ind1]-features_loc[ind2],2);
    }
    if(mse/2<pow((spa_spe[0] / 2.0), 2))
    {
        int ind1 = in_w+num_sample*in_h;
        int ind2 = in_h+num_sample*in_w;
        adj_sample_spa[ind1]=1;
        adj_sample_spa[ind2]=1;
    }
    mse = 0;
    for(int i=0;i<band;i++)
    {
        int ind1 = i+band*in_h;
        int ind2 = i+band*in_w;
        mse = mse +pow(features[ind1]-features[ind2],2);
    }
    mse = exp(-sqrt(mse/band));
    if(mse>spa_spe[1] )
    {
        int ind1 = in_w+num_sample*in_h;
        int ind2 = in_h+num_sample*in_w;
        adj_sample_spe[ind1]=mse;
        adj_sample_spe[ind2]=mse;
    }

}


void sample_forward_ongpu(float const *features,float const *features_loc, int num_sample, int band, float const *spa_spe, float *adj_sample_spa, float *adj_sample_spe)
{
    int size = num_sample*num_sample;

    //int spa_window=spa_spe[0];
    //int spe_threshold=spa_spe[1];
    //printf("t4");
    cudaError_t err;
    sample_forward_kernel<<<cuda_gridsize(size), BLOCK>>>(size, features, features_loc, num_sample, band, spa_spe, adj_sample_spa, adj_sample_spe);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#ifdef __cplusplus
}
#endif
