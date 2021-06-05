/*************************************************************************
    > File Name: convergence.h
    > Author: ZhangHeng
    > Mail: zhanghenglab@gmail.com
*/

#ifndef	CUDA_ERROR_CHECK_CUH
#define CUDA_ERROR_CHECK_CUH

#include <string>
#include <sstream>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDAErrorCheck(err) { CUDAAssert((err), __FILE__, __LINE__); }
inline void CUDAAssert( cudaError_t err, const char *file, int line )
{
   if ( err != cudaSuccess )
   {
	  std::ostringstream errStream;
	  errStream << "CUDAAssert: " << cudaGetErrorString(err) << " " << file << " " << line << "\n";
      throw std::runtime_error(errStream.str());
   }
}

#endif
