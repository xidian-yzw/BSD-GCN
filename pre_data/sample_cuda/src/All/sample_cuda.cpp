#include <THC/THC.h>
#include "sample_cuda_kernel.h"

#include <torch/extension.h>
extern THCState *state;

std::vector<torch::Tensor> sample_forward(torch::Tensor features, torch::Tensor features_loc, torch::Tensor spa_spe) {
  TORCH_CHECK(features.device().is_cuda(), "features must be a CUDA tensor");
  TORCH_CHECK(features_loc.device().is_cuda(), "features_loc must be a CUDA tensor");
  TORCH_CHECK(spa_spe.device().is_cuda(), "spa_spe must be a CUDA tensor");
  int num_sample   = features.size(0);
  int band = features.size(1);
  //std::cout<<"test1"<<num_sample<<"  "<<band;
  auto adj_sample_spa = torch::zeros({num_sample,num_sample},features.options());
  auto adj_sample_spe = torch::zeros({num_sample,num_sample},features.options());
  //std::cout<<"test";
  sample_forward_ongpu(features.data_ptr<float>(),features_loc.data_ptr<float>(), num_sample, band, spa_spe.data_ptr<float>(), adj_sample_spa.data_ptr<float>(), adj_sample_spe.data_ptr<float>());

  return {adj_sample_spa,adj_sample_spe};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sample_forward, "sample_forward_cuda");
}