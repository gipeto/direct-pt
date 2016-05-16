/*
*  Basic CUDA based triangle mesh path tracer.
*  For background info, see http://raytracey.blogspot.co.nz/2015/12/gpu-path-tracing-tutorial-2-interactive.html
*  Based on CUDA ray tracing code from http://cg.alexandra.dk/?p=278
*  Copyright (C) 2015  Sam Lapere
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*/

#pragma once


#include "cutil_math.h"  // required for float3 vector math
#include <curand.h>
#include <curand_kernel.h>
#include "CudaAPIHelper.h"

namespace cuTracer {

	using namespace cuApi;


#define M_PI 3.14159265359f


	// hardcoded camera position
	__device__ float3 firstcamorig = { 50, 52, 295.6 };


	struct Ray {
		float3 orig;	// ray origin
		float3 dir;		// ray direction	
		__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
	};

	enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance(), only DIFF used here

	// SPHERES

	struct Sphere {

		float rad;				// radius 
		float3 pos, emi, col;	// position, emission, color 
		Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)

		__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

			// Ray/sphere intersection
			// Quadratic formula required to solve ax^2 + bx + c = 0 
			// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
			// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

			float3 op = pos - r.orig;  // 
			float t, epsilon = 0.01f;
			float b = dot(op, r.dir);
			float disc = b*b - dot(op, op) + rad*rad; // discriminant
			if (disc<0) return 0; else disc = sqrtf(disc);
			return (t = b - disc)>epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
		}
	};

	// TRIANGLES

	// the classic ray triangle intersection: http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
	// for an explanation see http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection

	__device__ float RayTriangleIntersection(const Ray &r,
		const float3 &v0,
		const float3 &edge1,
		const float3 &edge2)
	{

		float3 tvec = r.orig - v0;
		float3 pvec = cross(r.dir, edge2);
		float  det = dot(edge1, pvec);

		det = __fdividef(1.0f, det);  // CUDA intrinsic function 

		float u = dot(tvec, pvec) * det;

		if (u < 0.0f || u > 1.0f)
			return -1.0f;

		float3 qvec = cross(tvec, edge1);

		float v = dot(r.dir, qvec) * det;

		if (v < 0.0f || (u + v) > 1.0f)
			return -1.0f;

		return dot(edge2, qvec) * det;
	}

	__device__ float3 getTriangleNormal(const cuArray<float4, false> & Triangles, const int triangleIndex){

		auto edge1 = Triangles[3 * triangleIndex + 1];
		auto edge2 = Triangles[3 * triangleIndex + 2];

		// cross product of two triangle edges yields a vector orthogonal to triangle plane
		auto trinormal = cross(make_float3(edge1.x, edge1.y, edge1.z), make_float3(edge2.x, edge2.y, edge2.z));
		trinormal = normalize(trinormal);

		return trinormal;
	}

	__device__ void intersectAllTriangles(const Ray& r, float& t_scene, int& triangle_id, const cuArray<float4, false> & Triangles, int& geomtype){

		for (auto i : make_cuRange(Triangles))
		{
			// the triangles are packed into the 1D texture using three consecutive float4 structs for each triangle, 
			// first float4 contains the first vertex, second float4 contains the first precomputed edge, third float4 contains second precomputed edge like this: 
			// (float4(vertex.x,vertex.y,vertex.z, 0), float4 (egde1.x,egde1.y,egde1.z,0),float4 (egde2.x,egde2.y,egde2.z,0)) 

			// i is triangle index, each triangle represented by 3 float4s in triangle_texture
			auto v0 = Triangles[3 * i];
			auto edge1 = Triangles[3 * i + 1];
			auto edge2 = Triangles[3 * i + 2];

			// intersect ray with reconstructed triangle	
			auto t = RayTriangleIntersection(r,
				make_float3(v0.x, v0.y, v0.z),
				make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z));

			// keep track of closest distance and closest triangle
			// if ray/tri intersection finds an intersection point that is closer than closest intersection found so far
			if (t < t_scene && t > 0.001)
			{
				t_scene = t;
				triangle_id = i;
				geomtype = 3;
			}
		}
	}


	// AXIS ALIGNED BOXES

	// helper functions
	inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
	inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
	inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
	inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

	struct Box {

		float3 min; // minimum bounds
		float3 max; // maximum bounds
		float3 emi; // emission
		float3 col; // colour
		Refl_t refl; // material type

		// ray/box intersection
		// for theoretical background of the algorithm see 
		// http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
		// optimised code from http://www.gamedev.net/topic/495636-raybox-collision-intersection-point/
		__device__ float intersect(const Ray &r) const {

			float epsilon = 0.001f; // required to prevent self intersection

			float3 tmin = (min - r.orig) / r.dir;
			float3 tmax = (max - r.orig) / r.dir;

			float3 real_min = minf3(tmin, tmax);
			float3 real_max = maxf3(tmin, tmax);

			float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
			float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

			if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
			else return 0;
		}

		// calculate normal for point on axis aligned box
		__device__ float3 Box::normalAt(float3 &point) {

			float3 normal = make_float3(0.f, 0.f, 0.f);
			float epsilon = 0.001f;

			if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
			else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
			else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
			else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
			else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
			else normal = make_float3(0, 0, 1);

			return normal;
		}
	};

	// scene: 9 spheres forming a Cornell box
	// small enough to fit in constant GPU memory
	__constant__ Sphere spheres[] = {
		// FORMAT: { float radius, float3 position, float3 emission, float3 colour, Refl_t material }
		// cornell box
		{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 1e5f
		{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Right 
		{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .9f, .9f, .9f }, SPEC }, //Back 
		{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { .9f, .9f, .9f }, SPEC }, //Front 
		{ 1e5f, { 50.0f, -1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Bottom 
		{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
		{ 16.5f, { 27.0f, 27.5f, 90.0f }, { 0.0f, 0.0f, 0.0f }, { 0.99f, 0.99f, 0.99f }, SPEC }, // small sphere 1
		{ 8.5f, { 73.0f, 8.5f, 78.0f }, { 0.0f, 0.f, .0f }, { 0.09f, 0.49f, 0.3f }, REFR }, // small sphere 2
		{ 600.0f, { 50.0f, 681.6f - .5f, 81.6f }, { 3.0f, 2.5f, 2.0f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light 12, 10 ,8

		//outdoor scene: radius, position, emission, color, material

		//{ 1600, { 3000.0f, 10, 6000 }, { 37, 34, 30 }, { 0.f, 0.f, 0.f }, DIFF },  // 37, 34, 30 // sun
		//{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f }, DIFF },  //  150, 75, 7.5 // sun 2
		//{ 10000, { 50.0f, 40.8f, -1060 }, { 0.0003, 0.01, 0.15 }, { 0.175f, 0.175f, 0.25f }, DIFF }, // sky
		//{ 100000, { 50.0f, -100000, 0 }, { 0.0, 0.0, 0 }, { 0.8f, 0.2f, 0.f }, DIFF }, // ground
		//{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f }, DIFF },  // horizon brightener
		//{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f }, DIFF }, // mountains
		//{ 82.5, { 30.0f, 180.5, 42 }, { 16, 12, 6 }, { .6f, .6f, 0.6f }, DIFF },  // small sphere 1
		//{ 12, { 115.0f, 10, 105 }, { 0.0, 0.0, 0.0 }, { 0.9f, 0.9f, 0.9f }, REFR },  // small sphere 2
		//{ 22, { 65.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f }, SPEC }, // small sphere 3
	};

	__constant__ Box boxes[] = {
		// FORMAT: { float3 minbounds,    float3 maxbounds,         float3 emission,    float3 colour,       Refl_t }
		{ { 5.0f, 0.0f, 70.0f }, { 45.0f, 11.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
		{ { 85.0f, 0.0f, 95.0f }, { 95.0f, 20.0f, 105.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
		{ { 75.0f, 20.0f, 85.0f }, { 105.0f, 22.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f }, DIFF },
	};


	__device__ inline bool intersect_scene(const Ray &r, float &t, int &sphere_id, int &box_id, int& triangle_id, const cuArray<float4, false> & Triangles, int &geomtype, const float3& bbmin, const float3& bbmax){


		float d = 1e21;
		float k = 1e21;
		float inf = t = 1e20;

		// SPHERES
		// intersect all spheres in the scene
		float numspheres = sizeof(spheres) / sizeof(Sphere);
		for (int i = int(numspheres); i--;)  // for all spheres in scene
			// keep track of distance from origin to closest intersection point
			if ((d = spheres[i].intersect(r)) && d < t){ t = d; sphere_id = i; geomtype = 1; }

		// BOXES
		// intersect all boxes in the scene
		float numboxes = sizeof(boxes) / sizeof(Box);
		for (int i = int(numboxes); i--;) // for all boxes in scene
			if ((k = boxes[i].intersect(r)) && k < t){ t = k; box_id = i; geomtype = 2; }

		// TRIANGLES
		Box scene_bbox; // bounding box around triangle meshes
		scene_bbox.min = bbmin;
		scene_bbox.max = bbmax;

		// if ray hits bounding box of triangle meshes, intersect ray with all triangles
		if (scene_bbox.intersect(r)){
			intersectAllTriangles(r, t, triangle_id, Triangles, geomtype);
		}

		// t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
		return t < inf;
	}


	// hash function to calculate new seed for each frame
	// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
	uint WangHash(uint a) {
		a = (a ^ 61) ^ (a >> 16);
		a = a + (a << 3);
		a = a ^ (a >> 4);
		a = a * 0x27d4eb2d;
		a = a ^ (a >> 15);
		return a;
	}

	// radiance function
	// compute path bounces in scene and accumulate returned color from each path sgment
	template<unsigned int bounces = 10>
	__device__ float3 radiance(Ray & r, curandState *randstate, const cuArray<float4, false> & Triangles, const float3& scene_aabb_min, const float3& scene_aabb_max){ // returns ray color

		// colour mask
		auto mask = make_float3(1.0f, 1.0f, 1.0f);
		// accumulated colour
		auto accucolor = make_float3(0.0f, 0.0f, 0.0f);

		for (auto b : make_cuRange(bounces)){  // iteration up to 4 bounces (instead of recursion in CPU code)

			// reset scene intersection function parameters
			float t = 100000; // distance to intersection 
			int sphere_id = -1;
			int box_id = -1;   // index of intersected sphere 
			int triangle_id = -1;
			int geomtype = -1;
			float3 f;  // primitive colour
			float3 emit; // primitive emission colour
			float3 x; // intersection point
			float3 n; // normal
			float3 nl; // oriented normal
			float3 d; // ray direction of next path segment
			Refl_t refltype;

			// intersect ray with scene
			// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
			if (!intersect_scene(r, t, sphere_id, box_id, triangle_id, Triangles, geomtype, scene_aabb_min, scene_aabb_max))
				return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

			// else: we've got a hit with a scene primitive
			// determine geometry type of primitive: sphere/box/triangle

			// if sphere:
			if (geomtype == 1){
				Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
				x = r.orig + r.dir*t;  // intersection point on object
				n = normalize(x - sphere.pos);		// normal
				nl = dot(n, r.dir) < 0 ? n : n * -1; // correctly oriented normal
				f = sphere.col;   // object colour
				refltype = sphere.refl;
				emit = sphere.emi;  // object emission
				accucolor += (mask * emit);
			}

			// if box:
			if (geomtype == 2){
				Box &box = boxes[box_id];
				x = r.orig + r.dir*t;  // intersection point on object
				n = normalize(box.normalAt(x)); // normal
				nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
				f = box.col;  // box colour
				refltype = box.refl;
				emit = box.emi; // box emission
				accucolor += (mask * emit);
			}

			// if triangle
			if (geomtype == 3){
				int tri_index = triangle_id;
				x = r.orig + r.dir*t;  // intersection point
				n = normalize(getTriangleNormal(Triangles, tri_index));  // normal 
				nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal

				// colour, refltype and emit value are hardcoded and apply to all triangles
				// no per triangle material support yet
				f = make_float3(0.9f, 0.4f, 0.1f);  // triangle colour
				refltype = REFR;
				emit = make_float3(0.0f, 0.0f, 0.0f);
				accucolor += (mask * emit);
			}

			// SHADING: diffuse, specular or refractive

			// ideal diffuse reflection (see "Realistic Ray Tracing", P. Shirley)
			if (refltype == DIFF){

				// create 2 random numbers
				float r1 = 2 * M_PI * curand_uniform(randstate);
				float r2 = curand_uniform(randstate);
				float r2s = sqrtf(r2);

				// compute orthonormal coordinate frame uvw with hitpoint as origin 
				float3 w = nl;
				float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
				float3 v = cross(w, u);

				// compute cosine weighted random ray direction on hemisphere 
				d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

				// offset origin next path segment to prevent self intersection
				x += nl * 0.03;

				// multiply mask with colour of object
				mask = mask*f;
			}

			// ideal specular reflection (mirror) 
			if (refltype == SPEC){

				// compute relfected ray direction according to Snell's law
				d = r.dir - 2.0f * n * dot(n, r.dir);

				// offset origin next path segment to prevent self intersection
				x += nl * 0.01f;

				// multiply mask with colour of object
				mask = mask*f;
			}

			// ideal refraction (based on smallpt code by Kevin Beason)
			if (refltype == REFR){

				bool into = dot(n, nl) > 0; // is ray entering or leaving refractive material?
				float nc = 1.0f;  // Index of Refraction air
				float nt = 1.5f;  // Index of Refraction glass/water
				float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
				float ddn = dot(r.dir, nl);
				float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

				if (cos2t < 0.0f) // total internal reflection 
				{
					d = reflect(r.dir, n); //d = r.dir - 2.0f * n * dot(n, r.dir);
					x += nl * 0.01f;
				}
				else // cos2t > 0
				{
					// compute direction of transmission ray
					float3 tdir = normalize(r.dir * nnt - n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))));

					float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
					float c = 1.f - (into ? -ddn : dot(tdir, n));
					float Re = R0 + (1.f - R0) * c * c * c * c * c;
					float Tr = 1 - Re; // Transmission
					float P = .25f + .5f * Re;
					float RP = Re / P;
					float TP = Tr / (1.f - P);

					// randomly choose reflection or transmission ray
					if (curand_uniform(randstate) < 0.25) // reflection ray
					{
						mask *= RP;
						d = reflect(r.dir, n);
						x += nl * 0.02f;
					}
					else // transmission ray
					{
						mask *= TP;
						d = tdir; //r = Ray(x, tdir); 
						x += nl * 0.0005f; // epsilon must be small to avoid artefacts
					}
				}
			}

			// set up origin and direction of next path segment
			r.orig = x;
			r.dir = d;
		}

		// add radiance up to a certain ray depth
		// return accumulated ray colour after all bounces are computed
		return accucolor;
	}

	template<unsigned int samps = 1>
	__global__ void render_kernel(cuArray<float3, false> accumbuffer, cuArray<float4, false> Triangles, int framenumber, uint hashedframenumber, float3 scene_bbmin, float3 scene_bbmax, uint2 size){   // float3 *gputexdata1, int *texoffsets


		for (auto threadId : make_cuRangeGrid(accumbuffer))
		{
			auto x = threadId % size.x;
			auto y = (threadId - x) / size.x;
			// create random number generator, see RichieSams blogspot
			curandState randState; // state of the random number generator, to prevent repetition
			curand_init(hashedframenumber + threadId, 0, 0, &randState);

			Ray cam(firstcamorig, normalize(make_float3(0, -0.042612, -1)));
			auto cx = make_float3(size.x * .5135 / size.y, 0.0f, 0.0f);  // ray direction offset along X-axis 
			auto cy = normalize(cross(cx, cam.dir)) * .5135; // ray dir offset along Y-axis, .5135 is FOV angle
			  

			int i = (size.y - y - 1)*size.x + x; // pixel index

			auto pixelcol = make_float3(0.0f, 0.0f, 0.0f); // reset to zero for every pixel	

			for (int s : make_cuRange(samps)){

				// compute primary ray direction
				float3 d = cx*((.25 + x) / size.x - .5) + cy*((.25 + y) / size.y - .5) + cam.dir;
				// normalize primary ray direction
				d = normalize(d);
				// add accumulated colour from path bounces

				Ray ray(cam.orig + d * 40, d);

				pixelcol += radiance(ray, &randState, Triangles, scene_bbmin, scene_bbmax);
			}       // Camera rays are pushed ^^^^^ forward to start in interior 

			// add pixel colour to accumulation buffer (accumulates all samples) 
			accumbuffer[i] += (pixelcol / samps);
		}

	}


	using OutputT = cuDX11SurfaceInterop<DXGI_FORMAT_B8G8R8A8_UNORM, D3D11_RESOURCE_DIMENSION_TEXTURE2D>;


	__global__ void OutputUpdateKernel(OutputT Output, cuArray<float3, false>  accumbuffer, uint2 Size, int framenumber)
	{

		for (auto x_idx : make_cuRangeGrid(accumbuffer))
		{
			auto x = x_idx % Size.x;
			auto y = (x_idx - x) / Size.x;

			// averaged colour: divide colour by the number of calculated frames so far
			auto tempcol = accumbuffer[x_idx] / framenumber;

			//Colour fcolour;
			float3 colour = make_float3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));

			auto Value = make_uchar4(static_cast<unsigned char>(powf(colour.x, 1 / 2.2f) * 255), static_cast<unsigned char>(powf(colour.y, 1 / 2.2f) * 255), static_cast<unsigned char>(powf(colour.z, 1 / 2.2f) * 255), 255);

			Output.PutSurface(Value, x, y);
		}


	}



	int frames = 0;

	volatile unsigned long long RefreshTexture = false;

	extern
		void Trace(cuArray<float4> & Triangles, cuArray<float3> & AccumulatedColor, cuStream & ComputationStream, float3 scene_aabbox_min, float3 scene_aabbox_max, uint2 Size)
	{

		frames++;

		// RAY TRACING:
		dim3 block(256, 1, 1);
		dim3 grid(512, 1, 1);

		// launch CUDA path tracing kernel, pass in a hashed seed based on number of frames
		render_kernel<2> << < grid, block, 0, ComputationStream >> >(AccumulatedColor, Triangles, frames, WangHash(frames), scene_aabbox_min, scene_aabbox_max, Size);  // launches CUDA render kernel from the host
		cuDebug(cudaPeekAtLastError());
		cuDebug(ComputationStream.synchronize());
		_InterlockedExchange(&RefreshTexture, 1);


	}


	extern
		void UpdateOutput(OutputT & Output, cuArray<float3> & AccBuffer, cuStream & RefreshStream, uint2 Size)
	{

		dim3 block(256, 1, 1);
		dim3 grid(512, 1, 1);

		if (1 == _InterlockedExchange(&RefreshTexture, 0))
		{
			cuDebug(cudaGraphicsMapResources(1, Output, RefreshStream));
			OutputUpdateKernel << <grid, block, 0, RefreshStream >> >(Output, AccBuffer, Size, frames);
			cuDebug(cudaPeekAtLastError());
			cuDebug(RefreshStream.synchronize());
			cuDebug(cudaGraphicsUnmapResources(1, Output, RefreshStream));
		}


	}


}









