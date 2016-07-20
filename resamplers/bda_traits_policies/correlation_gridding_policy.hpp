#pragma once
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "correlation_gridding_trait.hpp"
namespace imaging_core{
  namespace resamplers{
    class grid_single_correlation{
    private:
      typedef single_correlation corrT;
    public:
      __device__ __host__ static corrT read_vis(size_t row, 
				       size_t chan, 
				       size_t corr_index,
				       const thrust::device_vector<thrust::complex<float>> & vis_array){
	//TODO: STUB  
	return corrT(thrust::complex<float>(1,2));
      }
    };
  }
}