
#pragma once

#include "CudaAPIHelper.h"
#include "WindowsAPIHelper.h"

#include "cutil_math.h"
#include <iostream>
#include <fstream>
#include <string>


namespace cuTracer
{
	using namespace winApi;
	using namespace cuApi;

	using OutputT = cuDX11SurfaceInterop<DXGI_FORMAT_B8G8R8A8_UNORM, D3D11_RESOURCE_DIMENSION_TEXTURE2D>;

	extern
		void Trace(cuArray<float4> &, cuArray<float3> &, cuStream &, float3, float3, uint2);

	extern
		void UpdateOutput(OutputT &, cuArray<float3> &, cuStream &, uint2);


	// helpers to load triangle data
	struct TriangleFace
	{
		int v[3]; // vertex indices
	};

	struct TriangleMesh
	{
		std::vector<float3> verts;
		std::vector<TriangleFace> faces;
		float3 bounding_box[2];
	};


	class PathTracer :
		public Thread<PathTracer, CREATE_SUSPENDED>,
		public cuDeviceConfigurator
	{

		cuStream        m_TransferStream;
		cuStream        m_ComputationStream;
		cuArray<float4> m_Triangles;
		cuArray<float3> m_AccumBuffer;
		uint2           m_ImageSize;

		OutputT         m_OutputSurface;


	public:

		PathTracer() :
			PathTracer(1920, 1080)
		{}

		PathTracer(unsigned int width, unsigned int height) :
			m_ImageSize({ width, height }),
			m_AccumBuffer(width * height)
		{}

		unsigned int GetWidth()
		{
			return m_ImageSize.x;
		}

		unsigned int GetHeight()
		{
			return m_ImageSize.y;
		}


		void OnThreadStart()
		{
			m_Triangles.setStream(m_TransferStream);
			//	LoadData();
		}


		void Run()
		{
			while (IsRunning())
			{
				Trace(m_Triangles, m_AccumBuffer, m_ComputationStream, scene_aabbox_min, scene_aabbox_max, m_ImageSize);
			}

		}


		template<typename T>
		bool RegisterOutput(T Resource)
		{
			return cudaSuccess == m_OutputSurface.RegisterResource(Resource);
		}

		bool UnregisterOutput()
		{
			cudaGraphicsUnmapResources(1, m_OutputSurface, 0);
			return cudaSuccess == m_OutputSurface.UnregisterResource();
		}

		void RefreshOutput()
		{
			UpdateOutput(m_OutputSurface, m_AccumBuffer, m_ComputationStream, m_ImageSize);
		}

		float3 scene_aabbox_min = {};
		float3 scene_aabbox_max = {};


		void LoadData()
		{

			TriangleMesh mesh1;
			//	TriangleMesh mesh2;


			loadObj("D:\\direct-pt\\direct-pt.Desktop\\teapot.obj", mesh1);
			//	loadObj("D:\\direct-pt\\direct-pt.Desktop\\bunny.obj", mesh2);

			// scalefactor and offset to position/scale triangle meshes
			float scalefactor1 = 8;
			//			float scalefactor2 = 300;  // 300
			float3 offset1 = make_float3(90, 22, 100);// (30, -2, 80);
			//			float3 offset2 = make_float3(30, -2, 80);

			std::vector<float4> triangles;

			for (unsigned int i = 0; i < mesh1.faces.size(); i++)
			{
				// make a local copy of the triangle vertices
				float3 v0 = mesh1.verts[mesh1.faces[i].v[0] - 1];
				float3 v1 = mesh1.verts[mesh1.faces[i].v[1] - 1];
				float3 v2 = mesh1.verts[mesh1.faces[i].v[2] - 1];

				// scale
				v0 *= scalefactor1;
				v1 *= scalefactor1;
				v2 *= scalefactor1;

				// translate
				v0 += offset1;
				v1 += offset1;
				v2 += offset1;

				// store triangle data as float4
				// store two edges per triangle instead of vertices, to save some calculations in the
				// ray triangle intersection test
				triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
				triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
				triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
			}

			// compute bounding box of this mesh
			mesh1.bounding_box[0] *= scalefactor1; mesh1.bounding_box[0] += offset1;
			mesh1.bounding_box[1] *= scalefactor1; mesh1.bounding_box[1] += offset1;

			//	for (unsigned int i = 0; i < mesh2.faces.size(); i++)
			//	{
			//		float3 v0 = mesh2.verts[mesh2.faces[i].v[0] - 1];
			//		float3 v1 = mesh2.verts[mesh2.faces[i].v[1] - 1];
			//		float3 v2 = mesh2.verts[mesh2.faces[i].v[2] - 1];
			//
			//		v0 *= scalefactor2;
			//		v1 *= scalefactor2;
			//		v2 *= scalefactor2;
			//
			//		v0 += offset2;
			//		v1 += offset2;
			//		v2 += offset2;
			//
			//		triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
			//		triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 1));
			//		triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
			//	}
			//
			//	mesh2.bounding_box[0] *= scalefactor2; mesh2.bounding_box[0] += offset2;
			//	mesh2.bounding_box[1] *= scalefactor2; mesh2.bounding_box[1] += offset2;

			m_Triangles = triangles;

			// compute scene bounding box by merging bounding boxes of individual meshes 
			scene_aabbox_min = mesh1.bounding_box[0];
			scene_aabbox_max = mesh1.bounding_box[1];
			//	scene_aabbox_min = fminf(scene_aabbox_min, mesh2.bounding_box[0]);
			//	scene_aabbox_max = fmaxf(scene_aabbox_max, mesh2.bounding_box[1]);

		}

		// read triangle data from obj file
		void loadObj(const std::string filename, TriangleMesh &mesh)
		{
			std::ifstream in(filename.c_str());

			if (!in.good())
			{
				std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
				system("PAUSE");
				exit(0);
			}

			char buffer[256], str[255];
			float f1, f2, f3;

			while (!in.getline(buffer, 255).eof())
			{
				buffer[255] = '\0';
				sscanf_s(buffer, "%s", str, 255);

				// reading a vertex
				if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)){
					if (sscanf_s(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
						mesh.verts.push_back(make_float3(f1, f2, f3));
					}
					else{
						std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
						exit(-1);
					}
				}

				// reading faceMtls 
				else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
				{
					TriangleFace f;
					int nt = sscanf_s(buffer, "f %d %d %d", &f.v[0], &f.v[1], &f.v[2]);
					if (nt != 3){
						std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
						exit(-1);
					}

					mesh.faces.push_back(f);
				}
			}

			// calculate the bounding box of the mesh
			mesh.bounding_box[0] = make_float3(1000000, 1000000, 1000000);
			mesh.bounding_box[1] = make_float3(-1000000, -1000000, -1000000);
			for (unsigned int i = 0; i < mesh.verts.size(); i++)
			{
				//update min and max value
				mesh.bounding_box[0] = fminf(mesh.verts[i], mesh.bounding_box[0]);
				mesh.bounding_box[1] = fmaxf(mesh.verts[i], mesh.bounding_box[1]);
			}

			std::cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << std::endl;
			std::cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z << ") max:"
				<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z << ")" << std::endl;
		}


	};


}