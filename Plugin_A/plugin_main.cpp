#define USP_PLUGIN_DLL   1
#include "UspPlugin.h"
#undef USP_PLUGIN_DLL   

#include <cstdio>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////

// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
    "__kernel void square(                                                  \n" \
    "   __global float* input,                                              \n" \
    "   __global float* output,                                             \n" \
    "   const unsigned int count)                                           \n" \
    "{                                                                      \n" \
    "   int i = get_global_id(0);                                           \n" \
    "   if(i < count)                                                       \n" \
    "       output[i] = input[i] * input[i];                                \n" \
    "}                                                                      \n" \
    "\n";

////////////////////////////////////////////////////////////////////////////////



/// <summary>Statically defined structure describing plug-in capabilities. </summary>

static bool dll_initialized = false;
static cl_context g_ctx;
static cl_device_id g_dev_id;
static char* g_path_to_dll = NULL;
static cl_kernel g_kernel;
static cl_program g_program;




static BuffSize gInSize;
static BuffSize gOutSize;
static cl_uint g_count;




PLUGIN_API void __cdecl GetPluginInfo(PluginInfo* info)
{
    info->UseOpenCL = 1;
    info->InCLMem = 1;
    info->OutCLMem = 1;
    info->NumInBuffers = 1;
    info->NumOutBuffers = 1;
}

PLUGIN_API int __cdecl InitializeCL( cl_context ctx, cl_device_id id, char* path_to_dll )
{
    int err;
    g_ctx = ctx;
    g_dev_id = id;
    g_path_to_dll = path_to_dll;

    g_program = clCreateProgramWithSource(g_ctx, 1, (const char **) & KernelSource, NULL, &err);
    if (!g_program)
    {
        printf("Error: Failed to create compute program!\n");
        return -1;
    }


    // Build the program executable
    //
    err = clBuildProgram(g_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(g_program, g_dev_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    //
    g_kernel = clCreateKernel(g_program, "square", &err);
    if (!g_kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    return 0;
}


PLUGIN_API int __cdecl Initialize( char* path_to_dll )
{
    return 0;
}

PLUGIN_API int __cdecl Cleanup(void)
{

    clReleaseProgram(g_program);
    clReleaseKernel(g_kernel);

    return 0;
}

PLUGIN_API int __cdecl SetParams(float* pfp, size_t nfp, int* pip, size_t nip)
{
    return 0;
}

PLUGIN_API int __cdecl SetInBufSize(BuffSize* buf, int bufnum)
{
    if (bufnum == 0) {
        gInSize = *buf;       // 1 buffer
        gOutSize = gInSize;   // Output is equal to input
    } else {
        return -1;
    }
    return 0;
}



PLUGIN_API int __cdecl Prepare(void)
{
    // This is typically the place to initialize internal buffers etc.
    
    g_count = (cl_uint) gInSize.width;
    return 0;
}

PLUGIN_API int __cdecl GetOutBufSize(BuffSize* buf, int bufnum)
{
    *buf = gOutSize;
    return 0;
}

PLUGIN_API int __cdecl ProcessCLIO(cl_mem* inbuf, size_t numin, cl_mem* outbuf, size_t numout, cl_command_queue commands, cl_event inEv, cl_event* outEv)
{

    int err = 0;
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    err  = clSetKernelArg(g_kernel, 0, sizeof(cl_mem), inbuf);
    err |= clSetKernelArg(g_kernel, 1, sizeof(cl_mem), outbuf);
    err |= clSetKernelArg(g_kernel, 2, sizeof(g_count), &g_count);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(g_kernel, g_dev_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    //
    global = g_count;
    err = clEnqueueNDRangeKernel(commands, g_kernel, 1, NULL, &global, &local, 1, &inEv, outEv);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }
    clFinish(commands);
    return 0;
}


PLUGIN_API int __cdecl ProcessMemIO(void* inbuf[], size_t numin, void* outbuf[], size_t numout)
{
    return 0;
}


