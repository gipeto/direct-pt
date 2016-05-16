#include "Precompiled.h"
#include "DesktopWindowHelper.h"
#include "cuTracer.h"


struct dpt_win : public DXDesktopWindow<dpt_win>
{
	void OnWindowLoading()
	{
		if (m_Tracer.cudaSupported())
		{
			m_Adapter = m_Tracer.GetAdapter(m_DxgiFactory.Get());
			m_Tracer.Start();
		}
	}
 
	void Draw()
	{

		if (m_Tracer.cudaSupported())
		{
			m_Tracer.RefreshOutput();
			m_Target->SetTransform(d2d::Matrix3x2F::Identity());

			auto size = m_Target->GetSize();
			D2D_RECT_F Rect = {};
			Rect.right = size.width;
			Rect.top = size.height;

			m_Target->DrawBitmap(m_BitMap.Get(), Rect, 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR);

		}
		else
		{
			D2D1_COLOR_F Color = { 1.0f, 1.0f, 1.0f, 1.0f };
			m_Target->Clear(Color);
			auto size = m_Target->GetSize();

			D2D_RECT_F Rect = { size.width / 2, size.height / 2, size.width / 4, size.height / 4 };

			m_Target->DrawRectangle(Rect, m_Brush.Get(), 3.0f);

		}

	}

	wrl::ComPtr<ID2D1SolidColorBrush> m_Brush;
	wrl::ComPtr<ID3D11Texture2D>      m_Texture;
	wrl::ComPtr<ID2D1Bitmap>          m_BitMap;

	cuTracer::PathTracer m_Tracer = cuTracer::PathTracer();

	void CreateDeviceResources()
	{
		D2D1_COLOR_F Color = { 1.0f, 0.0f, 1.0f, 1.0f };
		HR(m_Target->CreateSolidColorBrush(Color, m_Brush.ReleaseAndGetAddressOf()));

		// Create D3D Texture to share with cuda

		wrl::ComPtr<ID3D11Device1> d3device;
		HR(m_SwapChain->GetDevice(__uuidof(d3device), reinterpret_cast<void **>(d3device.GetAddressOf())));

		D3D11_TEXTURE2D_DESC texDesc = {};
		texDesc.Width = m_Tracer.GetWidth();
		texDesc.Height = m_Tracer.GetHeight();
		texDesc.MipLevels = 1;
		texDesc.ArraySize = 1;
		texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
		texDesc.SampleDesc = { 1, 0 };
		texDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

		HR(d3device->CreateTexture2D(&texDesc, NULL, m_Texture.ReleaseAndGetAddressOf()));

		// Register texture with cuda
		m_Tracer.RegisterOutput(m_Texture.Get());

		// Get Dxgi surface from texture and use it to share content with a D2D bitmap
		wrl::ComPtr<IDXGISurface2> DxgiSurface;
		m_Texture.As(&DxgiSurface);

		DXGI_SURFACE_DESC dxgiDesc = {};
		DxgiSurface->GetDesc(&dxgiDesc);

		D2D1_BITMAP_PROPERTIES  bitmapDesc = {};;
		bitmapDesc.pixelFormat = D2D1::PixelFormat(dxgiDesc.Format, D2D1_ALPHA_MODE_IGNORE);

		HR(m_Target->CreateSharedBitmap(__uuidof(DxgiSurface), DxgiSurface.Get(), &bitmapDesc, m_BitMap.ReleaseAndGetAddressOf()));


	}

	void ReleaseDeviceResources()
	{
		m_Brush.Reset();
		m_Tracer.UnregisterOutput();
		m_BitMap.Reset();
		m_Texture.Reset();
	}

	~dpt_win()
	{
		m_Tracer.Stop();
		// Add a wrapper around the raw handle
		::WaitForSingleObjectEx(m_Tracer.Get(), INFINITE, TRUE);
	}

};


int __stdcall wWinMain(HINSTANCE module, HINSTANCE, PWSTR, int)
{

	dpt_win window;

	if (window.Create(module, L"window", L"title"))
	{
		window.Run();
	}

	return 0;

}