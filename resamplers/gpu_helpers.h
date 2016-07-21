#include <cuda.h>
#include <cufft.h>
#include <sstream>
/****************************************************************************************************************
 CUDA error handling macros
****************************************************************************************************************/
#define cudaSafeCall(value) {                                                                                   \
        cudaError_t _m_cudaStat = value;                                                                                \
        if (_m_cudaStat != cudaSuccess) {                                                                               \
		std::ostringstream msg;\
		msg << "Error" << cudaGetErrorString(value) << " at line " << __LINE__ << " in file " << __FILE__ << std::endl;\
		throw std::runtime_error(msg.str());\
        } }

inline void __cufftSafeCall( uint32_t err, const char *file, const int line ){
        if ( CUFFT_SUCCESS != err ){
                fprintf( stderr, "cufftSafeCall() failed at %s:%i\n", file, line);
                exit( -1 );
        }
        return;
}