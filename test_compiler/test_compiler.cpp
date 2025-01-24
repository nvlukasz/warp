#include <cudaTypedefs.h>
#include <cuda_runtime_api.h>

#include <nvrtc.h>

#include <stdio.h>
#include <dlfcn.h>
#include <vector>

//
// Build command:
//   /usr/local/cuda/bin/nvcc test_compiler.cpp -o test_compiler -lnvrtc_static -lnvrtc-builtins_static -lnvptxcompiler_static
//

#define check_cu(code) (check_cu_result(code, __FUNCTION__, __FILE__, __LINE__))
#define check_nvrtc(code) (check_nvrtc_result(code, __FILE__, __LINE__))

#if CUDA_VERSION < 12000
static PFN_cuGetProcAddress_v11030 pfn_cuGetProcAddress;
#else
static PFN_cuGetProcAddress_v12000 pfn_cuGetProcAddress;
#endif
static PFN_cuDriverGetVersion_v2020 pfn_cuDriverGetVersion;
static PFN_cuGetErrorName_v6000 pfn_cuGetErrorName;
static PFN_cuGetErrorString_v6000 pfn_cuGetErrorString;
static PFN_cuInit_v2000 pfn_cuInit;
static PFN_cuDeviceGet_v2000 pfn_cuDeviceGet;
static PFN_cuDevicePrimaryCtxRetain_v7000 pfn_cuDevicePrimaryCtxRetain;
static PFN_cuCtxSetCurrent_v4000 pfn_cuCtxSetCurrent;
static PFN_cuCtxSynchronize_v2000 pfn_cuCtxSynchronize;
static PFN_cuModuleLoadDataEx_v2010 pfn_cuModuleLoadDataEx;
static PFN_cuModuleUnload_v2000 pfn_cuModuleUnload;
static PFN_cuModuleGetFunction_v2000 pfn_cuModuleGetFunction;
static PFN_cuLaunchKernel_v4000 pfn_cuLaunchKernel;

bool check_cu_result(CUresult result, const char* func, const char* file, int line)
{
    if (result == CUDA_SUCCESS)
        return true;

    const char* errString = NULL;
    if (pfn_cuGetErrorString)
        pfn_cuGetErrorString(result, &errString);

    if (errString)
        fprintf(stderr, "CUDA error %u: %s (in function %s, %s:%d)\n", unsigned(result), errString, func, file, line);
    else
        fprintf(stderr, "CUDA error %u (in function %s, %s:%d)\n", unsigned(result), func, file, line);

    return false;
}

bool check_nvrtc_result(nvrtcResult result, const char* file, int line)
{
    if (result == NVRTC_SUCCESS)
        return true;

    const char* error_string = nvrtcGetErrorString(result);
    fprintf(stderr, "Warp NVRTC compilation error %u: %s (%s:%d)\n", unsigned(result), error_string, file, line);
    return false;
}

// Get versioned driver entry point. The version argument should match the function pointer type.
// For example, to initialize PFN_cuCtxCreate_v3020 use version 3020.
static bool get_driver_entry_point(const char* name, int version, void** pfn)
{
    if (!pfn_cuGetProcAddress || !name || !pfn)
        return false;

#if CUDA_VERSION < 12000
    CUresult r = pfn_cuGetProcAddress(name, pfn, version, CU_GET_PROC_ADDRESS_DEFAULT);
#else
    CUresult r = pfn_cuGetProcAddress(name, pfn, version, CU_GET_PROC_ADDRESS_DEFAULT, NULL);
#endif

    if (r != CUDA_SUCCESS)
    {
        fprintf(stderr, "Warp CUDA error: Failed to get driver entry point '%s' (CUDA error %u)\n", name, unsigned(r));
        return false;
    }

    return true;
}

#define DRIVER_ENTRY_POINT_ERROR driver_entry_point_error(__FUNCTION__)

static CUresult driver_entry_point_error(const char* function)
{
    fprintf(stderr, "Warp CUDA error: Function %s: a suitable driver entry point was not found\n", function);
    return (CUresult)cudaErrorCallRequiresNewerDriver; // this matches what cudart would do
}

CUresult cuDriverGetVersion_f(int* version)
{
    return pfn_cuDriverGetVersion ? pfn_cuDriverGetVersion(version) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGetErrorName_f(CUresult result, const char** pstr)
{
    return pfn_cuGetErrorName ? pfn_cuGetErrorName(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuGetErrorString_f(CUresult result, const char** pstr)
{
    return pfn_cuGetErrorString ? pfn_cuGetErrorString(result, pstr) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuInit_f(unsigned int flags)
{
    return pfn_cuInit ? pfn_cuInit(flags) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDeviceGet_f(CUdevice *dev, int ordinal)
{
    return pfn_cuDeviceGet ? pfn_cuDeviceGet(dev, ordinal) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxSetCurrent_f(CUcontext ctx)
{
    return pfn_cuCtxSetCurrent ? pfn_cuCtxSetCurrent(ctx) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuCtxSynchronize_f()
{
    return pfn_cuCtxSynchronize ? pfn_cuCtxSynchronize() : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuDevicePrimaryCtxRetain_f(CUcontext* ctx, CUdevice dev)
{
    return pfn_cuDevicePrimaryCtxRetain ? pfn_cuDevicePrimaryCtxRetain(ctx, dev) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleLoadDataEx_f(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues)
{
    return pfn_cuModuleLoadDataEx ? pfn_cuModuleLoadDataEx(module, image, numOptions, options, optionValues) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleUnload_f(CUmodule hmod)
{
    return pfn_cuModuleUnload ? pfn_cuModuleUnload(hmod) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuModuleGetFunction_f(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    return pfn_cuModuleGetFunction ? pfn_cuModuleGetFunction(hfunc, hmod, name) : DRIVER_ENTRY_POINT_ERROR;
}

CUresult cuLaunchKernel_f(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra)
{
    return pfn_cuLaunchKernel ? pfn_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra) : DRIVER_ENTRY_POINT_ERROR;
}


int main()
{
    printf("Hello\n");

    static void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
    if (hCudaDriver == NULL) {
        // WSL and possibly other systems might require the .1 suffix
        hCudaDriver = dlopen("libcuda.so.1", RTLD_NOW);
        if (hCudaDriver == NULL) {
            fprintf(stderr, "Warp CUDA error: Could not open libcuda.so.\n");
            return false;
        }
    }
    pfn_cuGetProcAddress = (PFN_cuGetProcAddress)dlsym(hCudaDriver, "cuGetProcAddress");

    if (!pfn_cuGetProcAddress)
    {
        fprintf(stderr, "Warp CUDA error: Failed to get function cuGetProcAddress\n");
        return false;
    }

    get_driver_entry_point("cuGetErrorString", 6000, &(void*&)pfn_cuGetErrorString);
    get_driver_entry_point("cuGetErrorName", 6000, &(void*&)pfn_cuGetErrorName);
    get_driver_entry_point("cuInit", 2000, &(void*&)pfn_cuInit);
    get_driver_entry_point("cuDeviceGet", 2000, &(void*&)pfn_cuDeviceGet);
    get_driver_entry_point("cuDevicePrimaryCtxRetain", 7000, &(void*&)pfn_cuDevicePrimaryCtxRetain);
    get_driver_entry_point("cuCtxSetCurrent", 4000, &(void*&)pfn_cuCtxSetCurrent);
    get_driver_entry_point("cuCtxSynchronize", 2000, &(void*&)pfn_cuCtxSynchronize);
    get_driver_entry_point("cuModuleLoadDataEx", 2010, &(void*&)pfn_cuModuleLoadDataEx);
    get_driver_entry_point("cuModuleUnload", 2000, &(void*&)pfn_cuModuleUnload);
    get_driver_entry_point("cuModuleGetFunction", 2000, &(void*&)pfn_cuModuleGetFunction);
    get_driver_entry_point("cuLaunchKernel", 4000, &(void*&)pfn_cuLaunchKernel);

    //
    // compile
    //

    // const char* src_path = "src/source1.cu";
    const char* src_path = "src/source2.cu";

    // read the source
    FILE* fp = fopen(src_path, "r");
    if (!fp)
    {
        fprintf(stderr, "*** Failed to open source file %s\n", src_path);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long flen = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    printf("Source size: %ld\n", flen);

    std::vector<char> source(flen + 1);
    fread(source.data(), 1, flen, fp);
    source[flen] = '\0';
    printf("----------\n");
    printf("%s", source.data());
    printf("----------\n");

    nvrtcProgram prog;
    check_nvrtc(nvrtcCreateProgram(&prog, source.data(), "source.cu", 0, NULL, NULL));

    // NVRTC compiler options
    std::vector<const char*> opts
    {
        "--std=c++17",
        "-I/work/src/cccl/libcudacxx/include",
        "-I/work/src/cccl/libcudacxx/include/cuda/std",
    };

    if (!check_nvrtc(nvrtcCompileProgram(prog, int(opts.size()), opts.data())))
    {
        size_t logSize = 0;
        check_nvrtc(nvrtcGetProgramLogSize(prog, &logSize));
        if (logSize > 1)
        {
            char *log = new char[logSize];
            check_nvrtc(nvrtcGetProgramLog(prog, log));
            printf("%s\n", log);
            delete[] log;
        }
        return 1;
    }

    size_t output_size;
    check_nvrtc(nvrtcGetPTXSize(prog, &output_size));
    printf("Output size: %u\n", unsigned(output_size));
    char *output = new char[output_size];
    check_nvrtc(nvrtcGetPTX(prog, output));

    check_nvrtc(nvrtcDestroyProgram(&prog));

    //
    // run
    //

    CUdevice dev;
    CUcontext ctx;
    CUmodule module;
    CUfunction kernel;

    check_cu(cuInit_f(0));
    check_cu(cuDeviceGet_f(&dev, 0));
    check_cu(cuDevicePrimaryCtxRetain_f(&ctx, dev));
    check_cu(cuCtxSetCurrent_f(ctx));

    CUjit_option options[2];
    void *option_vals[2];
    char error_log[8192] = "";
    unsigned int log_size = 8192;
    // Set up loader options
    // Pass a buffer for error message
    options[0] = CU_JIT_ERROR_LOG_BUFFER;
    option_vals[0] = (void*)error_log;
    // Pass the size of the error buffer
    options[1] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
    option_vals[1] = (void*)(size_t)log_size;

    if (!check_cu(cuModuleLoadDataEx_f(&module, output, 2, options, option_vals)))
    {
        fprintf(stderr, "Warp error: Loading CUDA module failed\n");
        // print error log if not empty
        if (*error_log)
            fprintf(stderr, "CUDA loader error:\n%s\n", error_log);
        return 1;
    }

    check_cu(cuModuleGetFunction_f(&kernel, module, "float_value_kernel"));
    printf("Kernel: %p\n", kernel);

    float param = 13.37f;
    std::vector<void*> params
    {
        &param,
    };

    check_cu(cuLaunchKernel_f(
        kernel,
        1, 1, 1,
        1, 1, 1,
        0, NULL,
        params.data(), NULL
    ));

    check_cu(cuCtxSynchronize_f());

    printf("Goodbye\n");

    return 0;
}
