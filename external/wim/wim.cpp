#include "wim.h"

#include <cuda_runtime.h>

#include <unordered_map>

#if defined(_WIN32)
    #define WIM_API __declspec(dllexport)
#else
    #define WIM_API __attribute__ ((visibility ("default")))
#endif

#define check_cuda(code) (wim::check_cuda_result(code, __FILE__, __LINE__))

// internal stuff
namespace wim
{

bool check_cuda_result(cudaError_t code, const char* file, int line)
{
    if (code == cudaSuccess)
        return true;

    fprintf(stderr, "WIM CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file, line);
    return false;
}

}

// API for Python bindings
extern "C"
{

WIM_API wim::Image* create_image_cpu(int width, int height, wim::Color** data_ret)
{
    wim::Color* data = new wim::Color[width * height];
    wim::Image* img = new wim::Image(width, height, data);

    if (data_ret)
        *data_ret = data;

    return img;
}

WIM_API void destroy_image_cpu(wim::Image* img)
{
    delete [] img->getData();
    delete img;
}

WIM_API wim::Image* create_image_cuda(int device, int width, int height, wim::Color** data_ret)
{
    if (!check_cuda(cudaSetDevice(device)))
        return nullptr;

    wim::Color* data = nullptr;
    if (!check_cuda(cudaMalloc(&data, width * height * sizeof(wim::Color))))
        return nullptr;
    if (!check_cuda(cudaMemset(data, 0, width * height * sizeof(wim::Color))))
        return nullptr;

    wim::Image img_h(width, height, data);

    wim::Image* img_d = nullptr;
    if (!check_cuda(cudaMalloc(&img_d, sizeof(wim::Image))))
        return nullptr;
    if (!check_cuda(cudaMemcpy(img_d, &img_h, sizeof(wim::Image), cudaMemcpyHostToDevice)))
        return nullptr;

    if (data_ret)
        *data_ret = data;

    return img_d;
}

WIM_API void destroy_image_cuda(int device, wim::Image* img_d)
{
    if (!img_d)
        return;

    if (!check_cuda(cudaSetDevice(device)))
        return;

    wim::Image img_h;
    if (!check_cuda(cudaMemcpy(&img_h, img_d, sizeof(wim::Image), cudaMemcpyDeviceToHost)))
        return;

    cudaFree(img_h.getData());
    cudaFree(img_d);
}

}
