#pragma once
#include "Precompiled.h"
#include <type_traits>
#include <tuple>
#include <vector>

#include <d2d1_1.h>
#include <d3d11_1.h>

#pragma comment(lib,"d2d1")
#pragma comment(lib,"d3d11")
#pragma comment(lib,"dxgi")

namespace d2d = D2D1;

namespace wrl
{

	using namespace Microsoft::WRL;
}


template<template <typename...> class C, typename T, int N, typename ... Args>
struct PackHelper
{
	typedef typename PackHelper<C, T, N - 1, T, Args...>::type type;
};

template<template <typename...> class C, typename T, typename ... Args>
struct PackHelper<C, T, 0, Args...>
{
	typedef C<Args...> type;
};


template<typename _T, int cnt, int N, _T Current, _T ... Next>
struct PackNthValue
{
	auto static constexpr Value = PackNthValue < _T, cnt + 1, N, Next... >::Value;
};

template<typename _T, int N, _T Current, _T ... Next>
struct PackNthValue<_T, N, N, Current, Next...>
{
	auto static constexpr Value = Current;
};



template< DWORD First, DWORD ... Rest>
struct MessageList
{

	typedef typename PackHelper<std::tuple, decltype(First), sizeof...(Rest)+1>::type MessageTuple;

	static auto constexpr GetAsTuple() -> MessageTuple
	{
		return MessageTuple(First, Rest...);
	}

	static auto GetAsVector()->std::vector<decltype(First)>
	{
		return{ First, Rest... };
	}

	template<int n>
	inline static constexpr DWORD Get()
	{
		return PackNthValue<DWORD, 0, n, First, Rest...>::Value;
	}


};



template<DWORD ClassStyle, DWORD WindowStyle>
struct WindowStyleTraits
{
	enum{ class_style = ClassStyle };
	enum{ window_style = WindowStyle };
};

template<typename T, typename WS = WindowStyleTraits< CS_HREDRAW | CS_VREDRAW, WS_BORDER | WS_VISIBLE> >
class DesktopWindowBase
{


	template<int CNT>
	struct MessageHelper
	{
		template<typename T>
		inline static auto ProcessMessage(T * pT, typename T::WMList::MessageTuple & MessageList, HWND window, UINT message, WPARAM wparam, LPARAM lparam) -> LRESULT
		{

			if (std::get<CNT - 1>(MessageList) == message)
			{
				// Need constexpr to get the value of the message at compile time
				// Use message position in message list if not available
#if _MSC_FULL_VER > 180040629 
				return pT->OnMessage<T::WMList::Get<CNT - 1>()>(window, wparam, lparam);
#else
				return pT->OnMessage<CNT - 1>(window, wparam, lparam);
#endif		
			}

			return MessageHelper<CNT - 1>::ProcessMessage(pT, MessageList, window, message, wparam, lparam);
		}

	};

	template<>
	struct MessageHelper<0>
	{
		template<typename T>
		inline static auto ProcessMessage(T * pT, typename T::WMList::MessageTuple & MessageList, HWND window, UINT message, WPARAM wparam, LPARAM lparam) -> LRESULT
		{
			return DefWindowProc(window, message, wparam, lparam);
		}
	};
protected:

	HWND m_hWnd = nullptr;

public:


	DesktopWindowBase()
	{}

	bool Create(HINSTANCE module, wchar_t * class_name, wchar_t * window_name)
	{
		WNDCLASS wc = {};
		wc.style = WS::class_style;
		wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
		wc.hInstance = module;
		wc.lpszClassName = class_name;

		wc.lpfnWndProc = [](HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)->LRESULT
		{

			if (WM_NCCREATE == message)
			{
				auto pcs = reinterpret_cast<CREATESTRUCT*>(lparam);
				auto pT = static_cast<T*>(pcs->lpCreateParams);
				pT->m_hWnd = hwnd;
				SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pcs->lpCreateParams));
				return TRUE;
			}
			else
			{
				auto pT = reinterpret_cast<T*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

				if (nullptr == pT)
				{
					return DefWindowProc(hwnd, message, wparam, lparam);
				}

				auto MsgList = T::WMList::GetAsTuple();
				return MessageHelper<std::tuple_size<decltype(MsgList)>::value>::ProcessMessage(pT, MsgList, hwnd, message, wparam, lparam);
			}

		};

		VERIFY(RegisterClass(&wc));


		auto style = WS::window_style;

		m_hWnd = CreateWindow(class_name, window_name, style,
			CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
			nullptr, nullptr, module, this);
		ASSERT(m_hWnd);

		return (nullptr != m_hWnd);

	}

	virtual void Run() = 0;

};



template<typename T, bool AnimationLoop = false, typename WS = WindowStyleTraits< CS_HREDRAW | CS_VREDRAW, WS_OVERLAPPEDWINDOW | WS_VISIBLE>  >
class DesktopWindowImpl : public DesktopWindowBase<T, WS>
{

protected:

	bool m_Visible = true;
	bool m_Minimized = false;


public:

	typedef MessageList<WM_DESTROY, WM_PAINT, WM_SIZE, WM_DISPLAYCHANGE, WM_GETMINMAXINFO, WM_ACTIVATE, WM_USER, WM_MOUSEMOVE> WMList;




	DesktopWindowImpl() = default;

	template<DWORD Message>
	inline  auto OnMessage(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
	{
		static_assert(false, "Missing message implementation");
		return 0;
	}

	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_DESTROY>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	// Map to message position in list if constexpr is not available 
	inline  auto OnMessage<0>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{
		PostQuitMessage(0);
		return 0;
	}


	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_PAINT>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<1>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{

		auto pT = static_cast<T*>(this);
		PAINTSTRUCT ps;
		VERIFY(BeginPaint(window, &ps));
		TRACE(L"Paint!\n");
		pT->Render();
		EndPaint(window, &ps);

		return 0;
	}

	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_SIZE>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<2>(HWND, WPARAM wparam, LPARAM)->LRESULT
#endif
	{
		auto pT = static_cast<T*>(this);
		m_Minimized = (SIZE_MINIMIZED == wparam);

		__if_exists (T::OnWindowResized)
		{
			pT->OnWindowResized();
		}

		return 0;
	}

	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_DISPLAYCHANGE>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<3>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{
		return 0;
	}

	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_GETMINMAXINFO>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<4>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{
		auto info = reinterpret_cast<MINMAXINFO *>(lparam);
		info->ptMinTrackSize.y = 200;
		return 0;
	}

	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_ACTIVATE>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<5>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{
		m_Visible = !HIWORD(wparam);
		return 0;
	}

	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_USER>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<6>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{
		auto pT = static_cast<T*>(this);

		__if_exists (T::OnWindowOccluded)
		{
			pT->OnWindowOccluded();
		}
		return 0;
	}



	template<>
#if _MSC_FULL_VER > 180040629
	inline  auto OnMessage<WM_MOUSEMOVE>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#else
	inline  auto OnMessage<7>(HWND window, WPARAM wparam, LPARAM lparam)->LRESULT
#endif
	{
		auto pT = static_cast<T*>(this);

		__if_exists (T::OnMouseMoved)
		{
			pT->OnMouseMoved(LOWORD(lparam), HIWORD(lparam));
		}
		return 0;
	}




	inline void Render()
	{}

	virtual void Run()
	{

		MSG  message = {};

		auto pT = static_cast<T*>(this);

		__if_exists (T::OnLoad)
		{
			pT->OnLoad();
		}

		if (AnimationLoop)
		{


			while (true)
			{
				if (m_Visible)
				{
					pT->Render();

					while (PeekMessage(&message, nullptr, 0, 0, PM_REMOVE))
					{
						DispatchMessage(&message);
					}
				}
				else
				{
					if (BOOL result = GetMessage(&message, 0, 0, 0))
					{
						if (-1 != result)
						{
							DispatchMessage(&message);
						}

					}

				}

				if (WM_QUIT == message.message)
				{
					break;
				}
			}
		}
		else
		{

			while (BOOL result = GetMessage(&message, 0, 0, 0))
			{
				if (-1 != result)
				{
					DispatchMessage(&message);
				}

			}
		}
	}



};

template<typename T>
class DXDesktopWindow : public DesktopWindowImpl<T, true>
{



	auto inline static CreateD2DFactory() -> wrl::ComPtr<ID2D1Factory1>
	{
		wrl::ComPtr<ID2D1Factory1> factory;

		D2D1_FACTORY_OPTIONS options = {};
#ifdef _DEBUG
		options.debugLevel = D2D1_DEBUG_LEVEL_INFORMATION;
#endif

		HR(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, options, factory.GetAddressOf()));
		return factory;

	}

	auto inline static CreateDevice(D3D_DRIVER_TYPE const type, const wrl::ComPtr<IDXGIAdapter1> & Adapter, wrl::ComPtr<ID3D11Device> & device) -> HRESULT
	{
		ASSERT(!device);
		UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if _DEBUG
		//flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

		auto hr = D3D11CreateDevice(nullptr,
			type,
			nullptr,
			flags,
			nullptr,
			0,
			D3D11_SDK_VERSION,
			device.GetAddressOf(),
			nullptr,
			nullptr);

		return hr;

	}

	inline auto CreateDevice(const wrl::ComPtr<IDXGIAdapter1> & Adapter)->wrl::ComPtr<ID3D11Device>
	{
		wrl::ComPtr<ID3D11Device> device;

		auto hr = CreateDevice(D3D_DRIVER_TYPE_HARDWARE, Adapter, device);

		if (DXGI_ERROR_UNSUPPORTED == hr)
		{
			hr = CreateDevice(D3D_DRIVER_TYPE_WARP, Adapter, device);
		}

		HR(hr);

		return device;

	}

	inline static auto CreateRenderTarget(wrl::ComPtr<ID2D1Factory1> const & factory,
		wrl::ComPtr<ID3D11Device> const & device)
		-> wrl::ComPtr<ID2D1DeviceContext>
	{

		ASSERT(factory);
		ASSERT(device);
		wrl::ComPtr<ID2D1DeviceContext>  target;
		wrl::ComPtr<IDXGIDevice> dxdevice;
		HR(device.As(&dxdevice));

		wrl::ComPtr<ID2D1Device> d2dDevice;


		HR(factory->CreateDevice(dxdevice.Get(), d2dDevice.GetAddressOf()));
		HR(d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, target.GetAddressOf()));

		return target;

	}

	inline static auto GetDxgiFactory(wrl::ComPtr<ID3D11Device> const & device) -> wrl::ComPtr<IDXGIFactory2>
	{
		ASSERT(device);
		wrl::ComPtr<IDXGIDevice> dxdevice;
		HR(device.As(&dxdevice));

		wrl::ComPtr<IDXGIAdapter> adapter;
		HR(dxdevice->GetAdapter(adapter.GetAddressOf()));

		wrl::ComPtr<IDXGIFactory2> factory;
		HR(adapter->GetParent(__uuidof(factory), reinterpret_cast<void **>(factory.GetAddressOf())));

		return factory;
	}

	inline static auto CreateSwapChainForWindow(wrl::ComPtr<ID3D11Device> device, HWND window)
		-> wrl::ComPtr<IDXGISwapChain1>
	{
		ASSERT(device);
		ASSERT(window);

		auto const factory = GetDxgiFactory(device);

		DXGI_SWAP_CHAIN_DESC1 props = {};
		props.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		props.SampleDesc.Count = 1;
		props.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
		props.BufferCount = 2;
		props.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

		wrl::ComPtr<IDXGISwapChain1> swapChain;
		HR(factory->CreateSwapChainForHwnd(device.Get(), window, &props, nullptr, nullptr, swapChain.GetAddressOf()));
		return swapChain;

	}


	inline static void CreateDeviceSwapChainBitmap(wrl::ComPtr<IDXGISwapChain1> swapchain, wrl::ComPtr<ID2D1DeviceContext> target)
	{
		ASSERT(swapchain);
		ASSERT(target);

		wrl::ComPtr<IDXGISurface> surface;
		HR(swapchain->GetBuffer(0, __uuidof(surface), reinterpret_cast<void **>(surface.GetAddressOf())));

		auto const props = d2d::BitmapProperties1(D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
			d2d::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));

		wrl::ComPtr<ID2D1Bitmap1> bitmap;
		HR(target->CreateBitmapFromDxgiSurface(surface.Get(),
			props,
			bitmap.GetAddressOf()));

		target->SetTarget(bitmap.Get());

	}


protected:

	wrl::ComPtr< ID2D1Factory1 >    m_Factory;
	wrl::ComPtr<ID2D1DeviceContext> m_Target;
	wrl::ComPtr<IDXGISwapChain1>    m_SwapChain;
	wrl::ComPtr<IDXGIFactory2>      m_DxgiFactory;
	wrl::ComPtr<IDXGIAdapter1>      m_Adapter;
	float m_dpi = 0;
	DWORD m_Occlusion = 0;


public:

	inline void OnLoad()
	{
		//	EventRegistrationToken token;
		//	HR(m_Window->add_SizeChanged(this, &token));
		float dpiY;
		m_Factory = CreateD2DFactory();
		m_Factory->GetDesktopDpi(&m_dpi, &dpiY);
		HR(CreateDXGIFactory1(__uuidof(m_DxgiFactory), reinterpret_cast<void **>(m_DxgiFactory.GetAddressOf())));
		auto pT = static_cast<T*>(this);


		__if_exists (T::OnWindowLoading)
		{
			pT->OnWindowLoading();
		}

		pT->CreateDeviceIndependentResources();
	}

	inline void OnWindowOccluded()
	{
		if (S_OK == m_SwapChain->Present(0, DXGI_PRESENT_TEST))
		{
			m_DxgiFactory->UnregisterOcclusionStatus(m_Occlusion);
			m_Occlusion = 0;
			m_Visible = true;

		}

	}


	//	inline void OnDpiChanged()
	//	{
	//		if (m_Target)
	//		{
	//			float dpi;
	//			auto hr = m_DisplayProperties->get_LogicalDpi(&dpi);
	//			m_Target->SetDpi(dpi, dpi);
	//			CreateDeviceSizeResources();
	//			Render();
	//		}
	//
	//	}


	inline void ResizeSwapChainBitmap()
	{
		auto pT = static_cast<T*>(this);

		ASSERT(m_Target);
		ASSERT(m_SwapChain);
		m_Target->SetTarget(nullptr);

		if (S_OK == m_SwapChain->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0))
		{
			CreateDeviceSwapChainBitmap(m_SwapChain, m_Target);
			pT->CreateDeviceSizeResources();
		}
		else
		{
			ReleaseDevice();
		}

	}


	inline void Render()
	{

		auto pT = static_cast<T*>(this);

		if (!m_Target)
		{

			auto device = CreateDevice(m_Adapter);
			m_Target = CreateRenderTarget(m_Factory, device);
			m_SwapChain = CreateSwapChainForWindow(device, m_hWnd);
			CreateDeviceSwapChainBitmap(m_SwapChain, m_Target);

			m_Target->SetDpi(m_dpi, m_dpi);

			pT->CreateDeviceResources();
			pT->CreateDeviceSizeResources();
		}

		m_Target->BeginDraw();
		pT->Draw();
		m_Target->EndDraw();

		auto const hr = m_SwapChain->Present(1, 0);

		if (S_OK == hr)
		{
		}
		else if (DXGI_STATUS_OCCLUDED == hr)
		{
			m_Visible = false;
			m_DxgiFactory->RegisterOcclusionStatusWindow(m_hWnd, WM_USER, &m_Occlusion);
		}
		else
		{
			ReleaseDevice();
		}


	}

	inline void OnWindowResized()
	{
		if (m_Target && !m_Minimized)
		{
			ResizeSwapChainBitmap();
			Render();
		}

	}

	inline void CreateDeviceIndependentResources()
	{}

	inline void CreateDeviceResources()
	{}

	inline void CreateDeviceSizeResources()
	{}

	inline void ReleaseDeviceResources()
	{}

	inline void ReleaseDevice()
	{
		auto pT = static_cast<T*>(this);

		m_Target.Reset();
		m_SwapChain.Reset();
		pT->ReleaseDeviceResources();
	}



};