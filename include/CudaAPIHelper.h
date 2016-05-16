
#pragma once
#include "Precompiled.h"
#include <type_traits>

namespace wrl
{
	using namespace Microsoft::WRL;
}

#include <d3d11_1.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>


#pragma comment(lib,"cudart_static.lib")

#include "cuErrorHandling.cuh"

namespace cuUtils
{

	namespace cuTraits
	{

		template<typename C>
		struct has_size
		{
			template <typename T>
			auto static check(T*) -> typename std::is_integral<decltype(std::declval<T const>().size())>::type;

			template <typename>
			auto static check(...)->std::false_type;

			typedef decltype(check<C>(0)) type;
			enum { value = type::value };
		};


		template<typename ... First>
		struct are_same_type : std::true_type
		{};

		template<typename First, typename Second, typename ... Rest>
		struct are_same_type<First, Second, Rest...> : std::conditional<std::is_same<First, Second>::value, typename are_same_type< Second, Rest...>::type, std::false_type>::type
		{};

		template<typename ... First>
		struct are_integrals : std::true_type
		{};

		template<typename First, typename...Rest>
		struct are_integrals<First, Rest...> : std::conditional<std::is_integral<First>::value, typename are_integrals<Rest...>::type, std::false_type>::type
		{};

		template<typename First, typename...Rest>
		struct pack_uniform_type
		{
			typedef typename std::enable_if<are_same_type<First, Rest...>::value, First>::type type;
		};

	}

}


namespace cuApi
{


	using namespace cuUtils;
	using namespace cuUtils::cuTraits;


	class cuDeviceConfigurator
	{

		bool m_CudaSupported = false;
		int m_DeviceIdx = -1;

	public:

		cuDeviceConfigurator()
		{

			// needs to be generalized with traits class, kiss for now.

			int deviceCount = -1;
			cuDebug(cudaGetDeviceCount(&deviceCount));
			ASSERT(0 < deviceCount);

			cudaDeviceProp propsWanted = {};
			propsWanted.major = 3;
			propsWanted.minor = 0;

			if (cudaSuccess != cudaChooseDevice(&m_DeviceIdx, &propsWanted))
			{
				m_CudaSupported = false;
				return;
			}

			cudaDeviceProp propsObtained = {};
			cuDebug(cudaGetDeviceProperties(&propsObtained, m_DeviceIdx));

			m_CudaSupported = (propsObtained.major > propsWanted.major) || ((propsObtained.major == propsWanted.major) && (propsObtained.minor >= propsWanted.minor));

			if (m_CudaSupported)
			{
				cuDebug(cudaSetDevice(m_DeviceIdx));
			}

		}

		bool cudaSupported()
		{
			return m_CudaSupported;
		}

		wrl::ComPtr<IDXGIAdapter1> GetAdapter(const wrl::ComPtr<IDXGIFactory2>  & Factory)
		{

			wrl::ComPtr<IDXGIAdapter1> Adapter;

			if (!m_CudaSupported)
			{
				return Adapter;
			}

			bool AdapterFound = false;

			for (unsigned int i = 0; !AdapterFound; ++i)
			{

				if (S_OK != Factory->EnumAdapters1(i, Adapter.ReleaseAndGetAddressOf()))
				{
					return nullptr;
				}

				int DeviceIdx = -1;
				if (cudaSuccess != cudaD3D11GetDevice(&DeviceIdx, Adapter.Get()))
				{
					return nullptr;
				}

				AdapterFound = (DeviceIdx == m_DeviceIdx);

			}

			return Adapter;

		}


		~cuDeviceConfigurator()
		{
			cudaDeviceSynchronize();
			cudaDeviceReset();
		}


	};



	template<typename T, bool Pinned = false, unsigned int flags = 0U>
	class cuHostAllocator : public std::allocator<T>
	{

	public:

		// add constexpr when possible
		bool isPinned()
		{
			return Pinned;
		}

		unsigned int getFlags()
		{
			return flags;
		}

		cuHostAllocator()
		{}

		template <class U>
		cuHostAllocator(const cuHostAllocator<U, Pinned, flags>&)
		{}

		template<typename U>
		struct rebind
		{
			typedef cuHostAllocator<U, Pinned, flags> other;
		};


		void deallocate(pointer _Ptr, size_type)
		{	// deallocate object at _Ptr, ignore size
			Pinned ? cuDebug(cudaFreeHost(_Ptr)) : ::operator delete(_Ptr);
		}

		pointer allocate(size_type _Count)
		{	// allocate array of _Count elements
			if (Pinned)
			{
				pointer pinned = nullptr;
				cuDebug(cudaHostAlloc(&pinned, _Count*sizeof(T), flags));
				return pinned;
			}

			return (std::_Allocate(_Count, (pointer)0));
		}

	};


	template<typename T, bool Pinned = false, unsigned int flags = cudaHostAllocDefault>
	using cuHostVector = typename std::vector<T, cuHostAllocator<T, Pinned, flags> >;

	template<typename T>
	using cuHostPinnedVector = typename std::vector<T, cuHostAllocator<T, true, cudaHostAllocDefault> >;

	auto static cuStreamDeleter = [](cudaStream_t Stream){cudaStreamDestroy(Stream); Stream = nullptr; };

	using cuUniqueStream = std::unique_ptr < std::remove_pointer_t<cudaStream_t>, decltype(cuStreamDeleter)>;
	using cuSharedStream = std::shared_ptr < std::remove_pointer_t<cudaStream_t>>;

	class cuStream
	{
		cuSharedStream m_Stream;

	public:

		cuStream(int flags = cudaStreamDefault)
		{
			cudaStream_t s;
			cuDebug(cudaStreamCreateWithFlags(&s, flags));
			m_Stream = cuSharedStream(s, cuStreamDeleter);
		}

		cuStream(const cuStream & Other) :
			m_Stream(Other.m_Stream)
		{}

		cuStream & operator=(const cuStream & Other)
		{
			ASSERT(this != &Other);
			m_Stream = Other.m_Stream;
			return *this;
		}

		explicit cuStream(cudaStream_t Stream) :
			m_Stream(Stream, cuStreamDeleter)
		{}

		cuStream(cuStream && Other) :
			m_Stream(std::move(Other.m_Stream))
		{}

		cuStream & operator=(cuStream && Other)
		{
			ASSERT(this != &Other);
			m_Stream = std::move(Other.m_Stream);
			return *this;
		}

		cudaStream_t get() const
		{
			return m_Stream.get();
		}

		operator cudaStream_t() const
		{
			return m_Stream.get();
		}

		inline cudaError synchronize()
		{
			return cudaStreamSynchronize(*this);
		}

		inline bool query()
		{
			return cudaSuccess == cudaStreamQuery(*this);
		}


	};


	template <typename T>
	class cuLoopRange
	{
		template<typename ElementT>
		class cuLoopRangeIterator
		{
		public:

			__host__ __device__  cuLoopRangeIterator(ElementT const value_, ElementT const step_) :
				value(value_), step(step_)  {}

			__host__ __device__  cuLoopRangeIterator(ElementT const value_) :
				cuLoopRangeIterator(value_, static_cast<ElementT>(1)){}

			__host__ __device__ __forceinline__ bool operator!= (cuLoopRangeIterator const &other) const
			{
				return !(*this == other);
			}

			__host__ __device__ __forceinline__ bool operator== (cuLoopRangeIterator const &other) const
			{
				return value >= other.value;
			}

			__host__ __device__ __forceinline__ ElementT const &operator*() const
			{
				return value;
			}

			__host__ __device__ __forceinline__ cuLoopRangeIterator &operator++()
			{
				value += step;
				return *this;
			}

		private:

			ElementT value;
			ElementT step;

		};

	public:

		__host__ __device__  cuLoopRange(T const from_, T const to_, T const step_) :
			from(from_), to(to_), step(step_){}

		__host__ __device__  cuLoopRange(T const from_, T const to_) :
			cuLoopRange(from_, to_, static_cast<T>(1)){}

		__host__ __device__  cuLoopRange(T const to_) :
			cuLoopRange(static_cast<T>(0), to_, static_cast<T>(1)){}

		__host__ __device__  cuLoopRangeIterator<T> begin() const
		{
			return{ from, step };
		}

		__host__ __device__  cuLoopRangeIterator<T> end() const
		{
			return{ to, step };
		}


	private:

		T from;
		T to;
		T step;

	};


	template<typename...Args>
	__host__ __device__ __forceinline__ auto make_cuRange(Args ... args)-> std::enable_if_t< are_integrals<Args...>::value, typename cuLoopRange<typename pack_uniform_type<Args...>::type> >
	{
		return cuLoopRange<typename pack_uniform_type<Args...>::type>(args...);
	}

	template<typename T>
	__host__ __device__ __forceinline__ auto make_cuRange(const T & Obj)->std::enable_if_t<has_size<T>::value, cuLoopRange< decltype(std::declval<T>().size())> >
	{
		using ElementT = decltype(std::declval<T>().size());
		return cuLoopRange<ElementT>(Obj.size());
	}

	template<typename T>
	__device__ __forceinline__ auto make_cuRangeGrid(T to) -> std::enable_if_t<std::is_integral<T>::value, cuLoopRange<T> >
	{
		return cuLoopRange<T>(static_cast<T>(threadIdx.x + blockIdx.x*blockDim.x), to, static_cast<T>(blockDim.x*gridDim.x));
	}

	template<typename T>
	__device__ __forceinline__ auto make_cuRangeGrid(T from, T to)->std::enable_if_t<std::is_integral<T>::value, cuLoopRange<T> >
	{
		return cuLoopRange<T>(static_cast<T>(from + threadIdx.x + blockIdx.x*blockDim.x), to, static_cast<T>(blockDim.x*gridDim.x));
	}


	template<typename T>
	__device__ __forceinline__ auto make_cuRangeGrid(const T & Obj)-> std::enable_if_t<has_size<T>::value, cuLoopRange< decltype(std::declval<T>().size())> >
	{
		using ElementT = decltype(std::declval<T>().size());
		return cuLoopRange<ElementT>(static_cast<ElementT>(threadIdx.x + blockIdx.x*blockDim.x), Obj.size(), static_cast<ElementT>(blockDim.x*gridDim.x));
	}


	// manage the device memory allocations and data transfert for data arrays
	// Managed has to be false for __global__ function parameters
	template<typename T, bool Managed = true>
	class cuArray
	{

		T *    m_pData = nullptr;
		size_t m_Size = 0;
		cudaStream_t m_Stream = nullptr;

	public:


		__host__ __device__ size_t size() const
		{
			return m_Size;
		}

		__host__ __device__ T* get() const
		{
			return m_pData;
		}


		__device__ __forceinline__ T operator[](unsigned int idx) const
		{
			return m_pData[idx];
		}

		__device__ __forceinline__ T & operator[](unsigned int idx)
		{
			return m_pData[idx];
		}


		// Host only methods

		cuArray() = default;
		cuArray(const cuArray & Other) = default;

		cuArray(cuArray && Other)
		{
			swap(*this, Other);
		}

		cuArray(const size_t N)
		{
			allocate<true>(N);
		}

		cuArray & operator= (cuArray Other)
		{
			swap(*this, Other);
			return *this;
		}

		cuArray(T * devicePtr, size_t N) :
			m_pData(devicePtr),
			m_Size(N)
		{}

		cuArray(cudaStream_t  Stream) :
			m_Stream(Stream)
		{}

		operator cuArray<T, false>() const
		{
			return cuArray<T, false>(m_pData, m_Size);
		}


		void setStream(cudaStream_t Stream)
		{
			m_Stream = Stream;
		}

		cudaStream_t getStream() const
		{
			return m_Stream;
		}


		template<typename A>
		cuArray & operator = (const std::vector<T, A> & hostVector)
		{
			if (hostVector.size() != m_Size)
			{
				allocate<false>(hostVector.size());
			}

			cuDebug(cudaMemcpyAsync(m_pData, hostVector.data(), m_Size*sizeof(T), cudaMemcpyHostToDevice, m_Stream));
			return *this;
		}


		template<bool Pinned = false, unsigned int flags = cudaHostAllocDefault>
		auto GetArray()->cuHostVector<T, Pinned, flags>
		{
			cuHostVector<T, Pinned, flags> hostVector(m_Size);
			cuDebug(cudaMemcpyAsync(hostVector.data(), m_pData, m_Size*sizeof(T), cudaMemcpyDeviceToHost, m_Stream));
			cudaStreamSynchronize(m_Stream);
			return hostVector;
		}

		void release()
		{
			if (!Managed)
			{
				return;
			}

			if (nullptr != m_pData)
			{
				cuDebug(cudaFree(m_pData));
				m_pData = nullptr;
			}

			m_Size = 0;
		}

		~cuArray()
		{
			release();
		}

	protected:

		friend void swap(cuArray & left, cuArray & right)
		{
			using std::swap;
			swap(left.m_pData, right.m_pData);
			swap(left.m_Size, right.m_Size);
			swap(left.m_Stream, right.m_Stream);
		}

	private:

		template<bool MemsetZero = true>
		void allocate(const size_t N)
		{
			ASSERT(N > 0);

			if (nullptr != m_pData)
			{
				release();
			}

			cuDebug(cudaMalloc<T>(&m_pData, sizeof(T) * N));

			if (MemsetZero)
			{
				cuDebug(cudaMemset(m_pData, 0, sizeof(T) * N));
			}

			m_Size = N;
		}


	};

	// Map an array to cuda texture memory
	template<int NX, int NY, int NZ, unsigned int arrayAllocFlag = cudaArrayDefault >
	class CudaTextureWrapper
	{

		template<typename T, unsigned int flag>
		struct TexureFetcher
		{

			static __forceinline__ __device__ T GetTexture(cudaTextureObject_t &texObj, float &x)
			{
				// static_assert ( NX >= 1 && NY == 0 && NZ == 0, "Not a 1D texture" )
				return tex1D<T>(texObj, x);
			}

			static __forceinline__ __device__ T GetTexture(cudaTextureObject_t &texObj, float &x, float &y)
			{
				return tex2D<T>(texObj, x, y);
			}

			static __forceinline__ __device__ T GetTexture(cudaTextureObject_t &texObj, float &x, float &y, float &z)
			{
				return tex3D<T>(texObj, x, y, z);
			}


		};


		template<typename T >
		struct TexureFetcher < T, cudaArrayLayered >
		{

			static __forceinline__ __device__ T GetTexture(cudaTextureObject_t &texObj, float &x, unsigned int &layer)
			{
				return tex1DLayered<T>(texObj, x, layer);
			}

			static __forceinline__ __device__ T GetTexture(cudaTextureObject_t &texObj, float &x, float &y, unsigned int &layer)
			{
				return tex2DLayered<T>(texObj, x, y, layer);
			}


		};

		template<typename T >
		struct TexureFetcher < T, cudaArrayCubemap >
		{

			static __forceinline__ __device__ T GetTexture(cudaTextureObject_t &texObj, float &x, float &y, float &z)
			{
				return texCubemap<T>(texObj, x, y, z);
			}

		};


		cudaArray * m_texData;
		cudaTextureObject_t m_TextureObject;

	public:

		__device__ __host__ CudaTextureWrapper() :
			m_texData(nullptr),
			m_TextureObject(0)
		{};


		template<typename T>
		void BindTexture(T * Data)
		{

			auto extent = make_cudaExtent(NX, NY, NZ);
			auto chDesc = cudaCreateChannelDesc <T>();

			cuDebug(cudaMalloc3DArray(&m_texData, &chDesc, extent, arrayAllocFlag));


			// Copy input data to device 3D Array
			cudaMemcpy3DParms copyParams = { 0 };
			copyParams.srcPtr = make_cudaPitchedPtr(static_cast<void *> (Data), extent.width * sizeof(T), extent.width, extent.height);
			copyParams.dstArray = m_texData;
			copyParams.extent = extent;
			copyParams.kind = cudaMemcpyHostToDevice;
			cuDebug(cudaMemcpy3D(&copyParams));


			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = m_texData;

			cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));

			texDesc.normalizedCoords = 0;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;
			texDesc.addressMode[2] = cudaAddressModeClamp;
			texDesc.readMode = cudaReadModeElementType;

			cuDebug(cudaCreateTextureObject(&m_TextureObject, &resDesc, &texDesc, NULL));


		}


		template<typename ReturnType, typename Par1T >
		__forceinline__ __device__ ReturnType GetTexture(Par1T x)
		{
			return TexureFetcher < ReturnType, arrayAllocFlag >::GetTexture(m_TextureObject, x);
		}

		template<typename ReturnType, typename Par1T, typename Par2T >
		__forceinline__ __device__ ReturnType GetTexture(Par1T x, Par2T yl)
		{
			return TexureFetcher < ReturnType, arrayAllocFlag >::GetTexture(m_TextureObject, x, yl);
		}

		template<typename ReturnType, typename Par1T, typename Par2T, typename Par3T >
		__forceinline__ __device__ ReturnType GetTexture(Par1T x, Par2T y, Par3T zl)
		{
			return TexureFetcher < ReturnType, arrayAllocFlag >::GetTexture(m_TextureObject, x, y, zl);
		}


		void UnBindTexture()
		{
			cudaDestroyTextureObject(m_TextureObject);
			cudaFreeArray(m_texData);
		}

	};


	template<DXGI_FORMAT>
	struct CudaTypeFromDxgiFormat;

	template<>
	struct CudaTypeFromDxgiFormat<DXGI_FORMAT_B8G8R8A8_UNORM>
	{
		typedef uchar4 type;
	};


	template<DXGI_FORMAT DxgiFormat, D3D11_RESOURCE_DIMENSION ResourceType >
	class cuDX11SurfaceInterop
	{

	public:

		cudaArray_t             m_SurfData = nullptr;
		cudaGraphicsResource_t  m_ImageResource = nullptr;
		cudaSurfaceObject_t     m_SurfaceObject = 0;


		template<typename T, D3D11_RESOURCE_DIMENSION>
		struct SurfaceFetcher;

		template<typename T>
		struct SurfaceFetcher<T, D3D11_RESOURCE_DIMENSION_TEXTURE1D>
		{
			static __forceinline__ __device__ T GetSurface(cudaSurfaceObject_t surfObj, int x, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
			{
				return surf1Dread<T>(surfObj, x, bMode);
			}

			static __forceinline__ __device__ void PutSurface(T & Value, cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
			{
				surf1Dwrite(Value, surfObj, x * sizeof(T), bMode);
			}

		};

		template<typename T>
		struct SurfaceFetcher<T, D3D11_RESOURCE_DIMENSION_TEXTURE2D>
		{

			static __forceinline__ __device__ T GetSurface(cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
			{
				return surf2Dread<T>(surfObj, x, y, bMode);
			}

			static __forceinline__ __device__ void PutSurface(T & Value, cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
			{
				surf2Dwrite(Value, surfObj, x * sizeof(T), y, bMode);
			}


		};

		template<typename T>
		struct SurfaceFetcher<T, D3D11_RESOURCE_DIMENSION_TEXTURE3D>
		{

			static __forceinline__ __device__ T GetSurface(cudaSurfaceObject_t surfObj, int x, int y, int z, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
			{
				return surf3Dread<T>(surfObj, x, y, z, bMode);
			}

			static __forceinline__ __device__ void PutSurface(T & Value, cudaSurfaceObject_t surfObj, int x, int y, int  z, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
			{
				surf3Dwrite(Value, surfObj, x * sizeof(T), y, z, bMode);
			}


		};


	public:


		using elementT = typename CudaTypeFromDxgiFormat<DxgiFormat>::type;

		cuDX11SurfaceInterop() = default;


		cudaError_t RegisterResource(ID3D11Resource * Resource, unsigned int flags = cudaGraphicsRegisterFlagsNone)
		{

			auto error = cudaGraphicsD3D11RegisterResource(&m_ImageResource, Resource, flags);
			if (cudaSuccess != error) return error;

			error = cudaGraphicsMapResources(1, &m_ImageResource, 0);
			if (cudaSuccess != error) return error;

			error = cudaGraphicsSubResourceGetMappedArray(&m_SurfData, m_ImageResource, 0, 0);
			if (cudaSuccess != error) return error;

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = m_SurfData;

			// Create the surface object
			error = cudaCreateSurfaceObject(&m_SurfaceObject, &resDesc);
			if (cudaSuccess != error) return error;

			return cudaGraphicsUnmapResources(1, &m_ImageResource, 0);

		}

		cudaError_t UnregisterResource()
		{
			cuDebug(cudaDestroySurfaceObject(m_SurfaceObject));
			return cudaGraphicsUnregisterResource(m_ImageResource);
		}

		operator cudaGraphicsResource_t *()
		{
			ASSERT(nullptr != m_ImageResource);
			return &m_ImageResource;
		}

		__forceinline__ __device__ elementT GetSurface(int x, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
		{
			return SurfaceFetcher < elementT, ResourceType >::GetSurface(m_SurfaceObject, x, bMode);
		}


		__forceinline__ __device__ elementT GetSurface(int x, int y, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
		{
			return SurfaceFetcher < elementT, ResourceType >::GetSurface(m_SurfaceObject, x, y, bMode);
		}


		__forceinline__ __device__ elementT GetSurface(int x, int y, int zl, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
		{
			return SurfaceFetcher < elementT, ResourceType >::GetSurface(m_SurfaceObject, x, y, zl, bMode);
		}

		__forceinline__ __device__ void PutSurface(elementT &Value, int x, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
		{
			SurfaceFetcher < elementT, ResourceType >::PutSurface(Value, m_SurfaceObject, x, bMode);
		}


		__forceinline__ __device__ void PutSurface(elementT &Value, int x, int y, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
		{
			SurfaceFetcher < elementT, ResourceType >::PutSurface(Value, m_SurfaceObject, x, y, bMode);
		}


		__forceinline__ __device__ void PutSurface(elementT &Value, int x, int y, int zl, cudaSurfaceBoundaryMode bMode = cudaBoundaryModeTrap)
		{
			SurfaceFetcher < elementT, ResourceType >::PutSurface(Value, m_SurfaceObject, x, y, zl, bMode);
		}

	};


}