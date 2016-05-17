#include "Precompiled.h"
#include "DesktopWindowHelper.h"
#include "cuTracer.h"

#include <dwrite.h>
#pragma comment(lib,"dwrite.lib")


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

	void OnMouseMoved(float x, float y)
	{
		D2D_POINT_2F Center = { x , y  };
		m_Brush->SetCenter(Center);
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
			DrawNotSupportedScreen();
		}

	}

	void DrawNotSupportedScreen()
	{
	
		D2D1_COLOR_F Color = { 1.0f, 1.0f, 1.0f, 1.0f };
		m_Target->Clear(Color);
		auto size = m_Target->GetSize();
		m_Brush->SetRadiusX(size.width / 2.0f);
		m_Brush->SetRadiusY(size.height / 2.0f);

		D2D_RECT_F Rect = {};
		Rect.right = size.width;
		Rect.top = size.height;

		m_Target->FillRectangle(Rect, m_Brush.Get());
		
		auto Margin = 30.f;
		auto TextSize = size;
		TextSize.width -= 2.f*Margin;
		TextSize.height-= 2.f*Margin;

		if (S_OK == m_TextLayout->SetMaxHeight(TextSize.height) && S_OK == m_TextLayout->SetMaxWidth(TextSize.width))
		{
			D2D_POINT_2F center = { Margin, Margin };
			m_Target->DrawTextLayout(center, m_TextLayout.Get(), m_TextBrush.Get());
		
		}

	
	}



	wrl::ComPtr<ID2D1RadialGradientBrush> m_Brush;
	wrl::ComPtr<ID2D1SolidColorBrush>     m_TextBrush;
	wrl::ComPtr<IDWriteTextLayout>        m_TextLayout;
	wrl::ComPtr<ID3D11Texture2D>          m_Texture;
	wrl::ComPtr<ID2D1Bitmap>              m_BitMap;


	cuTracer::PathTracer m_Tracer = cuTracer::PathTracer();

	void CreateDeviceIndependentResources()
	{
		wrl::ComPtr<IDWriteFactory> dwFactory;
		HR(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(dwFactory), reinterpret_cast<IUnknown**>(dwFactory.GetAddressOf())));

		wrl::ComPtr<IDWriteTextFormat> textFormat;
		HR(dwFactory->CreateTextFormat(L"Arial", nullptr, DWRITE_FONT_WEIGHT_NORMAL, DWRITE_FONT_STYLE_NORMAL, DWRITE_FONT_STRETCH_NORMAL, 50.f, L"", textFormat.GetAddressOf()));

		const wchar_t * text = L"It looks like Cuda is not supported on your machine....\n\nStill you can enjoy this beautiful Direct2D radial gradient";

		HR(dwFactory->CreateTextLayout(text, wcslen(text), textFormat.Get(), 100.0f, 100.0f, m_TextLayout.ReleaseAndGetAddressOf()));
	
	
	}



	void CreateDeviceResources()
	{
		// Brush DrawNotSupportedScreen


		D2D1_COLOR_F Black = { 0.0f, 0.0f, 0.0f, 0.5f };
		D2D1_COLOR_F White = { 1.0f, 1.0f, 1.0f, 1.0f };
		D2D1_COLOR_F Green = { 0.20f, 0.85f, 0.10f, 1.0f };
		D2D1_GRADIENT_STOP Stops[] = { { 0.0f, White }, { 1.0f, Green } };

		wrl::ComPtr<ID2D1GradientStopCollection> collection;
		HR(m_Target->CreateGradientStopCollection(Stops, _countof(Stops), collection.GetAddressOf()));

		D2D1_RADIAL_GRADIENT_BRUSH_PROPERTIES props = {};
		HR(m_Target->CreateRadialGradientBrush(props, collection.Get(), m_Brush.ReleaseAndGetAddressOf()));

		HR(m_Target->CreateSolidColorBrush(Black, m_TextBrush.ReleaseAndGetAddressOf()));
		
		// Create D3D Texture to share with cuda

		wrl::ComPtr<ID3D11Device1> d3device;
		HR(m_SwapChain->GetDevice(__uuidof(d3device), reinterpret_cast<void **>(d3device.GetAddressOf())));

		D3D11_TEXTURE2D_DESC texDesc = {};
		texDesc.Width     = m_Tracer.GetWidth();
		texDesc.Height    = m_Tracer.GetHeight();
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
		if (m_Tracer.IsRunning())
		{
			m_Tracer.Stop();
			// Add a wrapper around the raw handle
			::WaitForSingleObjectEx(m_Tracer.Get(), INFINITE, TRUE);
		}
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