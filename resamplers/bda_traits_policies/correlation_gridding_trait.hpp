#pragma once
#include <thrust/complex.h>
namespace imaging_core{
  namespace resamplers{
    struct single_correlation{
      thrust::complex<float> m_corr1;
      __device__ __host__ single_correlation(const thrust::complex<float> & corr1){
	  this->m_corr1 = corr1;
      }
    };
    struct duel_correlation{
      thrust::complex<float> m_corr1,m_corr2;
    };
    struct quad_correlation{
      thrust::complex<float> m_corr1,m_corr2,m_corr3,m_corr4;
    };
  }
}