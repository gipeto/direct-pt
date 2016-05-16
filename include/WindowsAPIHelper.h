#include "Precompiled.h"


namespace winApi
{

	template<typename T, DWORD dwCreationFlags = 0, DWORD  dwStackSize = 0x20000 >
	class Thread
	{

	public:

		Thread()
		{

			m_Handle = ::CreateThread(0,
				dwStackSize,
				[](LPVOID lpParam)-> DWORD WINAPI
			{
				auto pT = static_cast<T*>(lpParam);
				pT->m_Running = true;

				__if_exists(T::OnThreadStart)
				{
					pT->OnThreadStart();
				}

				pT->Run();

				__if_exists(T::OnThreadTerminate)
				{
					pT->OnThreadTerminate();
				}


				return 0;
			},
				this,
				dwCreationFlags,
				&m_ThreadId
				);
		}


		DWORD Resume()
		{
			return ::ResumeThread(m_Handle);
		}

		DWORD Start()
		{
			return Resume();
		}

		void Stop()
		{
			m_Running = false;
		}

		int GetPriority()
		{
			return ::GetThreadPriority(m_Handle);
		}

		bool SetPriority(int Value)
		{
			return ::SetThreadPriority(m_Handle, Value) != FALSE;
		}

		DWORD GetThreadId()
		{
			return m_ThreadId;
		}


		bool IsRunning()
		{
			return m_Running;
		}

		~Thread()
		{}

		HANDLE Get()
		{
			return m_Handle;
		}

	private:

		HANDLE m_Handle = nullptr;
		DWORD  m_ThreadId;
		volatile bool   m_Running = false;
	};


}