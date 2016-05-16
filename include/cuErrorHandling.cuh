#pragma once



namespace cuUtils
{

#ifndef __CUDA_ARCH__
#define CUARCH 0
#else
#define CUARCH __CUDA_ARCH__
#endif


	__host__ __device__ __forceinline__ cudaError_t Debug(
		cudaError_t     error,
		const char*     filename,
		int             line)
	{

		if (error)
		{
#if (CUARCH == 0)
			OutputDebugPrintf(_T("CUDA error %d [%hs, %d]: %hs\n"), error, filename, line, cudaGetErrorString(error));
			ASSERT(false);

#elif (CUARCH >= 200)

			OutputDebugPrintf(_T("CUDA error %d [block %d, thread %d, %hs, %d]\n"), error, blockIdx.x, threadIdx.x, filename, line);
#endif
		}

		return error;
	}


#ifdef _DEBUG

#define cuDebug(e) Debug((e), __FILE__, __LINE__)
#define cuDebugExit(e) if (Debug((e), __FILE__, __LINE__)) { exit(1); }

#else

#define cuDebug(e) (e)
#define cuDebugExit(e) (e)

#endif


}

