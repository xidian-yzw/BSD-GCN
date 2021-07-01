#ifndef _BOTTOMPOOLING_CUDA_KERNEL
#define _BOTTOMPOOLING_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif
void sample_split_forward_ongpu( float const *select_index, float const *features,float const *features_loc, int min_num_sample, int num_sample, int band, float const *spa_spe,float *adj_sample_spa, float *adj_sample_spe);

#ifdef __cplusplus
}
#endif

#endif