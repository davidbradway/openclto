/// <summary> Plugin library for host. </summary>
#define USP_PLUGIN_DLL 1
#include "UspPlugin.h"
#include "UspDebug.h"
#include "Parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Macros for size calculations
#define CEIL(num, div) (num + div -1)/div
#define ROUND_UP(num, mul)  (((num) + (mul) - 1) / (mul))*mul

#if _MSC_VER
#define snprintf _snprintf
#endif

// Struct containing the many elements used by the OpenCL
static struct Glob{
    cl_context ctx;             // OpenCL context. Sent by the host application
    cl_device_id device;        // The device id is also sent by the host application
    char srcOpenCL[1024];
    char * program_source;

    cl_program prog;
	
	cl_kernel split_kernel, combine_kernel, std_dev_kernel, vel_est_kernel, arctan_kernel, to_vel_est_kernel, to_arctan_kernel, maxabsval_kernel, maxabsval2_kernel;

	cl_event event0, event1, event2, event3, event4, event5, event6, event7, event8;

	// for split kernel
	cl_mem Z;
	cl_mem Z2; // don't care?
	cl_mem L;
	cl_mem R;

	cl_mem std_dev_sum1_real; //could pack as float2 std_dev_sum1
	cl_mem std_dev_sum1_imag;
	cl_mem std_dev_sum2;
	cl_mem std_dev;

	// For vel_est (output) and arctan (input) kernels
	cl_mem temp0;
	cl_mem temp_re;
	cl_mem temp_im;

	cl_mem to_vel_est_sum12_re_im;

	// For maxabsval
	cl_mem scratch, result1, result2;
	
	// For maxabsval2
	cl_mem maximum;

	// for combine kernel
	cl_mem outbufZ;
	cl_mem outbufZX; //don't care?
	cl_mem outbufX;
	
	BuffSize inSize[1], outSize[2];

    size_t dataLen, length;

	size_t split_globWrkSize;    	size_t split_locWrkSize;
	size_t globWrkSize;             size_t locWrkSize;
	size_t std_dev_globWrkSize;    	size_t std_dev_locWrkSize;
	size_t arctan_globWrkSize;      size_t arctan_locWrkSize;
	size_t to_vel_est_globWrkSize;	size_t to_vel_est_locWrkSize;
	size_t to_arctan_globWrkSize;	size_t to_arctan_locWrkSize;
	size_t maxabsval_globWrkSize;	size_t maxabsval_locWrkSize;
	size_t maxabsval2_globWrkSize;	size_t maxabsval2_locWrkSize;
	size_t combine_globWrkSize;    	size_t combine_locWrkSize;

	//Parameter struct sent by the host application
	ParamStruct params; 

	bool memAllocated;
}glob;

const float PI = static_cast<float>(3.1415927);

/// <summary> Load OpenCL source file and save content.
/// The source file is loaded from path saved in struct.
/// Path is set in the function InitializeCL and therefore 
/// this is only intended for use inside that function.
/// The function returns boolean false if file is not found 
/// or whole file is not read, else returns boolean succes.
/// </summary>
bool LoadOpenCLSrc()
{
    bool success = true;

	// Step 06: Open kernel file
    FILE *file = fopen(glob.srcOpenCL, "rb");
    if (!file)
	{
		printf("Could not open or find file\n");
		return false;
	}
    fseek(file, 0, SEEK_END);
    size_t len = ftell(file);
	rewind(file);
    if (glob.program_source != NULL){
        free (glob.program_source);
        glob.program_source = NULL;
    }
    glob.program_source = (char*) malloc(len+1);
	glob.program_source[len] = '\0';

	// Step 06: Read kernel file
    size_t count = fread(glob.program_source, sizeof(char), len, file);
    fclose(file);
	//printf("count: %d, len: %d.\n",count,len);
    if (count != len) {
		printf("Not whole program was read\n");
        return false;
    }
    return success;
}

/// <summary> A clean up function.
/// The OpenCL objects are released with relevant OpenCL functions.
/// Allocated memory is also freed.
/// This function must not be called until after ProcessCL
/// Returns an OpenCL error number if releasing fails, else returns 0.
/// </summary>
PLUGIN_API int  Cleanup(void)
{
	// Step 13: Free objects
	int err = CL_SUCCESS;
    err |= clReleaseProgram(glob.prog);
    
	err |= clReleaseKernel(glob.split_kernel);
	err |= clReleaseKernel(glob.vel_est_kernel);
	err |= clReleaseKernel(glob.std_dev_kernel);
	err |= clReleaseKernel(glob.arctan_kernel);
	err |= clReleaseKernel(glob.to_vel_est_kernel);
	err |= clReleaseKernel(glob.to_arctan_kernel);
	err |= clReleaseKernel(glob.maxabsval_kernel);
	err |= clReleaseKernel(glob.maxabsval2_kernel);
	err |= clReleaseKernel(glob.combine_kernel);

	err |= clReleaseEvent(glob.event0);
	err |= clReleaseEvent(glob.event1);
	err |= clReleaseEvent(glob.event2);
	err |= clReleaseEvent(glob.event3);
	err |= clReleaseEvent(glob.event4);
	err |= clReleaseEvent(glob.event5);
	err |= clReleaseEvent(glob.event6);
	err |= clReleaseEvent(glob.event7);
	err |= clReleaseEvent(glob.event8);

	// for split kernel
	err |= clReleaseMemObject(glob.Z);
	err |= clReleaseMemObject(glob.Z2); // don't care?
	err |= clReleaseMemObject(glob.L);
	err |= clReleaseMemObject(glob.R);

	err |= clReleaseMemObject(glob.std_dev_sum1_real); //could pack as float2
	err |= clReleaseMemObject(glob.std_dev_sum1_imag);
	err |= clReleaseMemObject(glob.std_dev_sum2);
	err |= clReleaseMemObject(glob.std_dev);
	err |= clReleaseMemObject(glob.temp0);
	err |= clReleaseMemObject(glob.temp_re);
	err |= clReleaseMemObject(glob.temp_im);
	err |= clReleaseMemObject(glob.to_vel_est_sum12_re_im);

	// for maxabsval kernel
	err |= clReleaseMemObject(glob.scratch);
	err |= clReleaseMemObject(glob.result1);
	err |= clReleaseMemObject(glob.result2);
	
	// for maxabsval2 kernel
	err |= clReleaseMemObject(glob.maximum);

	// for combine kernel
	err |= clReleaseMemObject(glob.outbufZ);
	err |= clReleaseMemObject(glob.outbufZX); //don't care?
	err |= clReleaseMemObject(glob.outbufX);

	if(err != CL_SUCCESS)return err;

    if (glob.program_source != 0) {
        free(glob.program_source);
        glob.program_source = NULL;
    }
    return 0;
}

/// <summary> Sets plugin info for OpenCL api.
/// Input argument is a pointer to a PluginInfo struct.
/// Relevant info is set in struct and nothing is returned.
/// @param info A pointer to a PluginInfo struct
/// </summary>
PLUGIN_API void  GetPluginInfo(PluginInfo* info)
{
	//printf("get plugin info\n");
    info->UseOpenCL = 1;
	info->InCLMem = 1;  //1 or 0
    info->OutCLMem = 1; //1 or 0
    info->NumInBuffers = 1;
    info->NumOutBuffers = 2;
}

/// <summary> Creates OpenCL program and initializes important OpenCL objects.
/// Sets the path to OpenCL program file, loads content of file and creates OpenCL program.
/// Then builds program and if this fails, debugging information is printed.
/// Afterwards an OpenCL event and kernels are created.
/// Function returns -1 or -2 respectively if setting file path fails or loading of file fails.
/// An OpenCL error code is returned if any OpenCL function call fails.
/// @param ctx An OpenCL context in which the program and kernels are to be created.
/// @param id An OpenCL device ID also for the program and kernels.
/// @param path_to_kernel_file A char pointer to dll path
/// </summary>
PLUGIN_API int  InitializeCL( cl_context ctx, cl_device_id id, char* path_to_module )
{
	printf("Path to module: %s \n", path_to_module);
    int err = 0;
    glob.ctx = ctx;
    glob.device = id;

	//Set path to OpenCL program file
	memset(glob.srcOpenCL, 0, sizeof(glob.srcOpenCL));
	if (0 > snprintf(glob.srcOpenCL, sizeof(glob.srcOpenCL), "%s\\%s\0", path_to_module, "scale.cl")){
		printf("Function: Initialize, Error in setting path\n");
        return - 1;
	}

	printf("OpenCL file: %s \n", glob.srcOpenCL);
	// Step 06: Read kernel file
	if (!LoadOpenCLSrc())
	{
		printf("Function: Initialize, Error in LoadOpenCLSrc()\n");
		return -2;
	}
	
	// Step 07: Create Kernel program from the read in source
    glob.prog = clCreateProgramWithSource(glob.ctx, 1, (const char **) & glob.program_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create program with source!\n");
		return err;
	}

	// Step 08: Build Kernel Program
	printf("Before build\n");
    err |= clBuildProgram(glob.prog, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(glob.prog, glob.device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
	printf("After build\n");

    // Step 09: Create OpenCL Kernels
    int glob_err = 0;
	int step = 0;

	glob.split_kernel      = clCreateKernel(glob.prog, "split",           &err); glob_err |= err; 
	glob.vel_est_kernel    = clCreateKernel(glob.prog, "velocity_est",    &err); glob_err |= err; 
	glob.std_dev_kernel    = clCreateKernel(glob.prog, "std_dev",         &err); glob_err |= err; 
	glob.arctan_kernel     = clCreateKernel(glob.prog, "arctan",          &err); glob_err |= err; 
	glob.to_vel_est_kernel = clCreateKernel(glob.prog, "to_velocity_est", &err); glob_err |= err; 
	glob.to_arctan_kernel  = clCreateKernel(glob.prog, "to_arctan",       &err); glob_err |= err; 
	glob.maxabsval_kernel  = clCreateKernel(glob.prog, "maxabsval",       &err); glob_err |= err; 
	glob.maxabsval2_kernel = clCreateKernel(glob.prog, "maxabsval2",      &err); glob_err |= err;
	glob.combine_kernel    = clCreateKernel(glob.prog, "combine",         &err); glob_err |= err;
    if (glob_err != CL_SUCCESS) return glob_err;
	
	// Create user event objects
	glob.event0 = clCreateUserEvent(glob.ctx, &err); 
	glob.event1 = clCreateUserEvent(glob.ctx, &err); 
	glob.event2 = clCreateUserEvent(glob.ctx, &err); 
	glob.event3 = clCreateUserEvent(glob.ctx, &err); 
	glob.event4 = clCreateUserEvent(glob.ctx, &err); 
	glob.event5 = clCreateUserEvent(glob.ctx, &err); 
	glob.event6 = clCreateUserEvent(glob.ctx, &err); 
	glob.event7 = clCreateUserEvent(glob.ctx, &err); 
	glob.event8 = clCreateUserEvent(glob.ctx, &err); 
	printf("end initialize\n");
	return 0;
}

/// <summary> Not needed in current implementation </summary>
PLUGIN_API int  Initialize( char* path_to_dll )
{
    return 0;
}

/// <summary>Sets parameters
/// Returns zero no matter what.
/// @param pfp A pointer to a Float array of parameters
/// @param nfp Integer value number of float parameters
/// @param pip A pointer to a Integer array of parameters
/// @param nip Integer value number of int parameters
/// </summary>
PLUGIN_API int  SetParams(float* pfp, size_t nfp, int* pip, size_t nip)
{
	// Set parameters in Global struct
	glob.params.emissions    = pip[ind_emissions];
	glob.params.nlines       = pip[ind_nlines];
	glob.params.nlinesamples = pip[ind_nlinesamples];
	glob.params.numb_avg     = pip[ind_numb_avg];
	glob.params.avg_offset   = pip[ind_avg_offset];
	glob.params.lag_axial    = pip[ind_lag_axial];
	glob.params.lag_TO       = pip[ind_lag_TO];
	glob.params.lag_acq      = pip[ind_lag_acq];
	
	glob.params.fs           = pfp[ind_fs];
	glob.params.f0           = pfp[ind_f0];
	glob.params.c            = pfp[ind_c];
	glob.params.fprf         = pfp[ind_fprf];
	glob.params.depth        = pfp[ind_depth];
	glob.params.lambda_X     = pfp[ind_lambda_X];
	return 0;
}

/// <summary>Sets size of input buffers.
/// Copies the info in BuffSize struct
/// This function uses information from parameters and 
/// therefore must not be called before SetParams.
/// Returns -1, if sampleType is not correct
/// @param buf A pointer to a BuffSize struct containing input buffer information
/// @param bufnum Integer value
/// </summary>
PLUGIN_API int  SetInBufSize(BuffSize* buf, int bufnum)
{
	//if (buf->sampleType != SAMPLE_FORMAT_INT16) return -1;
	if (buf->sampleType != SAMPLE_FORMAT_INT16X2) return -1;
    glob.inSize[bufnum]  = *buf;
	return 0;
}

/// <summary>Prepares OpenCL kernels for execution.
/// Calculates global work size based on hardcoded local work size for the two kernels.
/// Then does some memory handling of intermediate buffers and creates buffers.
/// At last the kernel arguments that doesn't change are set.
/// This function must not be called before InitializeCL
/// Returns an OpenCL error number if any OpenCL function fails, else returns 0.
/// </summary>
PLUGIN_API int  Prepare(void)
{
	printf("start prepare\n");
	// Set parameters and arguments for stand-alone DLL in UseCase
	float scale   = static_cast<float>(glob.params.c*glob.params.fprf/(4.0*PI*glob.params.f0*glob.params.lag_axial)/glob.params.lag_acq);
	float k_axial = static_cast<float>(glob.params.c*glob.params.fprf/(2.0*PI*4.0*glob.params.f0)/glob.params.lag_acq);
	float k_trans = static_cast<float>(glob.params.fprf*glob.params.c*glob.params.lambda_X/(2.0*glob.params.fs*glob.params.depth*2.0*PI*2.0*glob.params.lag_TO*glob.params.lag_acq));
	int Nsamples  = glob.params.nlines * glob.params.nlinesamples;

    // This is typically the place to initialize internal buffers etc.
    // The right way is to keep track if buffers have been allocated
    // and to release them if reallocation is needed
    cl_int err = CL_SUCCESS;

	// TODO: fix this so it doesn't take 6ms. Avoid writing to memory.
	// Split kernel
	glob.split_locWrkSize = 64;       glob.split_globWrkSize = (size_t)(ROUND_UP(glob.params.nlines*glob.params.emissions,glob.split_locWrkSize));
	//glob.split_locWrkSize = 64;       glob.split_globWrkSize = (size_t)(ROUND_UP(glob.params.nlinesamples*4*glob.params.nlines*glob.params.emissions,glob.split_locWrkSize));
	printf("split:            global work size: %d, local work size: %d\n",glob.split_globWrkSize,glob.split_locWrkSize);

	// Standard deviation kernel
	glob.std_dev_locWrkSize = 64;    glob.std_dev_globWrkSize = (size_t)(ROUND_UP(CEIL(Nsamples,8),glob.std_dev_locWrkSize));
	//printf("std_dev:          global work size: %d, local work size: %d\n",glob.std_dev_globWrkSize,glob.std_dev_locWrkSize);

	// velocity_est kernel
	glob.locWrkSize = 64;            glob.globWrkSize = (size_t)(ROUND_UP(Nsamples,glob.locWrkSize));
	//printf("velocity_est:     global work size: %d, local work size: %d\n",glob.globWrkSize,glob.locWrkSize);

	// Arctan kernel
	glob.arctan_locWrkSize = 64;     glob.arctan_globWrkSize = (size_t)(ROUND_UP(Nsamples,glob.arctan_locWrkSize));
	//printf("arctan:           global work size: %d, local work size: %d\n",glob.arctan_globWrkSize,glob.arctan_locWrkSize);

	// to_velocity_est kernel
	glob.to_vel_est_locWrkSize = 64; glob.to_vel_est_globWrkSize = (size_t)(ROUND_UP(Nsamples,glob.to_vel_est_locWrkSize));
	//printf("to_velocity_est:  global work size: %d, local work size: %d\n",glob.to_vel_est_globWrkSize,glob.to_vel_est_locWrkSize);

	// TO_Arctan kernel
	glob.to_arctan_locWrkSize = 64;  glob.to_arctan_globWrkSize = (size_t)(ROUND_UP(Nsamples,glob.to_arctan_locWrkSize));
	//printf("to_arctan:        global work size: %d, local work size: %d\n",glob.to_arctan_globWrkSize,glob.to_arctan_locWrkSize);

	// maxabsval kernel
	glob.maxabsval_locWrkSize = 64;  glob.maxabsval_globWrkSize = (size_t)(ROUND_UP(Nsamples,glob.maxabsval_locWrkSize));
	printf("maxabsval:          global work size: %d, local work size: %d\n",glob.maxabsval_globWrkSize,glob.maxabsval_locWrkSize);

	// maxabsval2 kernel
	glob.maxabsval2_locWrkSize = 1;  glob.maxabsval2_globWrkSize = (size_t)(ROUND_UP(1,glob.maxabsval2_locWrkSize));
	printf("maxabsval2:       global work size: %d, local work size: %d\n",glob.maxabsval2_globWrkSize,glob.maxabsval2_locWrkSize);

	// Combine kernel
	glob.combine_locWrkSize = 64;    glob.combine_globWrkSize = (size_t)(ROUND_UP(Nsamples,glob.combine_locWrkSize));
	//printf("combine:          global work size: %d, local work size: %d\n",glob.combine_globWrkSize,glob.combine_locWrkSize);

	// Buffer memory checking and handling for split kernel
	if (glob.Z  != 0) { clReleaseMemObject(glob.Z);  glob.Z  = 0; }
	if (glob.Z2 != 0) { clReleaseMemObject(glob.Z2); glob.Z2 = 0; }
	if (glob.R  != 0) { clReleaseMemObject(glob.R);  glob.R  = 0; }
	if (glob.L  != 0) { clReleaseMemObject(glob.L);  glob.L  = 0; }

	// Buffer memory checking and handling for std deviation kernel
	if (glob.std_dev_sum1_real != 0) { clReleaseMemObject(glob.std_dev_sum1_real); glob.std_dev_sum1_real = 0; } //could pack as float2
	if (glob.std_dev_sum1_imag != 0) { clReleaseMemObject(glob.std_dev_sum1_imag); glob.std_dev_sum1_imag = 0; }
	if (glob.std_dev_sum2      != 0) { clReleaseMemObject(glob.std_dev_sum2);      glob.std_dev_sum2 = 0;      }
	if (glob.std_dev           != 0) { clReleaseMemObject(glob.std_dev);           glob.std_dev = 0;           }
	
	// Buffer memory checking and handling for vel_est/arctan kernels
	if (glob.temp0 != 0) { clReleaseMemObject(glob.temp0); glob.temp0 = 0; }
	if (glob.temp_re  != 0) { clReleaseMemObject(glob.temp_re);  glob.temp_re = 0;  }
	if (glob.temp_im  != 0) { clReleaseMemObject(glob.temp_im);  glob.temp_im= 0;  }
	
	// Buffer memory checking and handling for to_vel_est/to_arctan kernels
	if(glob.to_vel_est_sum12_re_im != 0){ clReleaseMemObject(glob.to_vel_est_sum12_re_im); glob.to_vel_est_sum12_re_im = 0; }

	// Buffer memory checking and handling for maxabsval kernel
	if (glob.scratch != 0) { clReleaseMemObject(glob.scratch); glob.scratch = 0; }
	if (glob.result1 != 0) { clReleaseMemObject(glob.result1); glob.result1 = 0; }
	if (glob.result2 != 0) { clReleaseMemObject(glob.result2); glob.result2 = 0; }
			
	// Buffer memory checking and handling for maxabsval2 kernel?
	if (glob.maximum != 0) { clReleaseMemObject(glob.maximum); glob.maximum = 0; }

	// Buffer memory checking and handling for combine kernel
	if (glob.outbufZ != 0) { clReleaseMemObject(glob.outbufZ);  glob.outbufZ = 0; }
	if (glob.outbufZX!= 0) { clReleaseMemObject(glob.outbufZX); glob.outbufZX= 0; } //don't care?
	if (glob.outbufX != 0) { clReleaseMemObject(glob.outbufX);  glob.outbufX = 0; }

	// Step 05: Create memory buffer objects
	glob.Z  = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*glob.params.emissions*sizeof(cl_float2), NULL, &err);
	glob.Z2 = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*glob.params.emissions*sizeof(cl_float2), NULL, &err);
	glob.L  = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*glob.params.emissions*sizeof(cl_float2), NULL, &err);
	glob.R  = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*glob.params.emissions*sizeof(cl_float2), NULL, &err); 

	// Buffer creation for std deviation kernel
	glob.std_dev_sum1_real   = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*sizeof(float), NULL, &err);  //could pack as float2
	glob.std_dev_sum1_imag   = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*sizeof(float), NULL, &err); 
	glob.std_dev_sum2        = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*sizeof(float), NULL, &err); 
	glob.std_dev             = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, sizeof(float), NULL, &err); 

	// Buffer creation for vel_est/arctan kernels
	glob.temp_re             = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.globWrkSize*sizeof(cl_float), NULL, &err);
	glob.temp_im             = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.globWrkSize*sizeof(cl_float), NULL, &err);

	// Buffer creation for to_vel_est/to_arctan kernels
	glob.to_vel_est_sum12_re_im = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.globWrkSize*sizeof(cl_float4), NULL, &err);

	// Buffer creation for arctan_kernel 
	glob.outbufZ     = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*sizeof(cl_float), NULL, &err); 

	// Buffer creation for to_arctan_kernel
	glob.outbufZX    = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*sizeof(cl_float), NULL, &err); //don't care?
	glob.outbufX     = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, glob.params.nlinesamples*glob.params.nlines*sizeof(cl_float), NULL, &err); 

	// Buffer creation for maxabsval_kernel
	glob.scratch = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, 64*sizeof(cl_float), NULL, &err);
	glob.result1 = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, ROUND_UP(glob.params.nlinesamples*glob.params.nlines,64)*sizeof(cl_float), NULL, &err); // 64 is local work size
	glob.result2 = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, ROUND_UP(glob.params.nlinesamples*glob.params.nlines,64)*sizeof(cl_float), NULL, &err); // 64 is local work size
	if (err != CL_SUCCESS)return err;

	// Buffer creation for maxabsval2_kernel
	glob.maximum = clCreateBuffer(glob.ctx, CL_MEM_READ_WRITE, 1*sizeof(cl_float), NULL, &err); //just need one
	if (err != CL_SUCCESS)return err;

	// Step 10: Set OpenCL kernel arguments	that don't change
	err |= clSetKernelArg(glob.std_dev_kernel,    5, sizeof(cl_int),   &glob.params.emissions);    
		
	err |= clSetKernelArg(glob.vel_est_kernel,    3, sizeof(cl_int),   &glob.params.emissions);   
	err |= clSetKernelArg(glob.vel_est_kernel,    4, sizeof(cl_int),   &Nsamples);
	
	err |= clSetKernelArg(glob.to_vel_est_kernel, 2, sizeof(cl_int),   &glob.params.lag_TO);       
	err |= clSetKernelArg(glob.to_vel_est_kernel, 3, sizeof(cl_int),   &glob.params.emissions);    
	err |= clSetKernelArg(glob.to_vel_est_kernel, 4, sizeof(cl_int),   &Nsamples);
	
	err |= clSetKernelArg(glob.to_arctan_kernel,  1, sizeof(cl_float), &k_axial);
	err |= clSetKernelArg(glob.to_arctan_kernel,  2, sizeof(cl_float), &k_trans);
	err |= clSetKernelArg(glob.to_arctan_kernel,  3, sizeof(cl_int),   &glob.params.numb_avg);     
	err |= clSetKernelArg(glob.to_arctan_kernel,  4, sizeof(cl_int),   &glob.params.avg_offset);   
    err |= clSetKernelArg(glob.to_arctan_kernel,  5, sizeof(cl_int),   &glob.params.nlinesamples); 
	if (err != CL_SUCCESS)return err;

	return 0;
}

/// <summary>Gets size of output buffer.
/// Copies the information on output buffer into passed BuffSize struct.
/// returns 0 no matter what.
/// @param buf A pointer to a BuffSize struct to contain copied information.
/// @param bufnum An integer that has no effect at the moment.
/// </summary>
PLUGIN_API int  GetOutBufSize(BuffSize* buf, int bufnum)
{
	glob.outSize[bufnum].sampleType = SAMPLE_FORMAT_INT8;
	glob.outSize[bufnum].width      = glob.params.nlinesamples;
	glob.outSize[bufnum].height     = glob.params.nlines;
	glob.outSize[bufnum].depth      = 1;
	glob.outSize[bufnum].widthLen   = glob.outSize[bufnum].width    * sizeof(signed char);
	glob.outSize[bufnum].heightLen  = glob.outSize[bufnum].widthLen * glob.outSize[bufnum].height;
	glob.outSize[bufnum].depthLen   = glob.outSize[bufnum].depth    * glob.outSize[bufnum].heightLen;
	*buf = glob.outSize[bufnum];
	//printf("outSize[%d].\n",bufnum);
	//printf(" sampleType: %d\n",glob.outSize[bufnum].sampleType);
	//printf(" width:      %d\n",glob.outSize[bufnum].width);
	//printf(" height:     %d\n",glob.outSize[bufnum].height);
	//printf(" depth: %d\n",glob.outSize[bufnum].depth);
	//printf(" widthLen:   %d\n",glob.outSize[bufnum].widthLen);
	//printf(" heightLen:  %d\n",glob.outSize[bufnum].heightLen);
	//printf(" depthLen:   %d\n",glob.outSize[bufnum].depthLen);
	return 0;
}

/// <summary>Executes OpenCL kernels program on GPU </summary>
PLUGIN_API int ProcessCLIO(cl_mem* inbuf, size_t numin, cl_mem* outbuf, size_t numout, cl_command_queue  clqueue, cl_event inEv, cl_event* outEv)
{
	float scale = static_cast<float>(glob.params.c*glob.params.fprf/(4.0*PI*glob.params.f0*glob.params.lag_axial)/glob.params.lag_acq);
	int Nsamples = glob.params.nlines*glob.params.nlinesamples;
	int N1 = glob.params.nlines*glob.params.emissions;
	int threads = Nsamples/64;

	cl_int err = CL_SUCCESS;
	// Step 10: Set OpenCL kernel arguments
	// Step 11: Execute OpenCL kernel in data parallel

	// Split kernel arguments
	err  = clSetKernelArg(glob.split_kernel,     0, sizeof(cl_mem), inbuf);
	err |= clSetKernelArg(glob.split_kernel,     1, sizeof(cl_int), &glob.params.nlinesamples);
	err |= clSetKernelArg(glob.split_kernel,     2, sizeof(cl_mem), &glob.Z);
	err |= clSetKernelArg(glob.split_kernel,     3, sizeof(cl_mem), &glob.Z2);
	err |= clSetKernelArg(glob.split_kernel,     4, sizeof(cl_mem), &glob.L);
	err |= clSetKernelArg(glob.split_kernel,     5, sizeof(cl_mem), &glob.R);
	err |= clSetKernelArg(glob.split_kernel,     6, sizeof(cl_int),	&N1);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.split_kernel,      1, NULL, &glob.split_globWrkSize,      &glob.split_locWrkSize,      1, &inEv,        &glob.event0);
	if (err != CL_SUCCESS)return err;
	//printf("after 1\n");

	// Standard deviation kernel arguments
	err  = clSetKernelArg(glob.std_dev_kernel,    0, sizeof(cl_mem), &glob.Z);
	err |= clSetKernelArg(glob.std_dev_kernel,    1, sizeof(cl_mem), &glob.std_dev_sum1_real); //could pack as float2
	err |= clSetKernelArg(glob.std_dev_kernel,    2, sizeof(cl_mem), &glob.std_dev_sum1_imag);
	err |= clSetKernelArg(glob.std_dev_kernel,    3, sizeof(cl_mem), &glob.std_dev_sum2);
	err |= clSetKernelArg(glob.std_dev_kernel,    4, sizeof(cl_int), &Nsamples);
	err |= clSetKernelArg(glob.std_dev_kernel,    6, sizeof(cl_mem), &glob.std_dev);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.std_dev_kernel,    1, NULL, &glob.std_dev_globWrkSize,    &glob.std_dev_locWrkSize,    1, &glob.event0, &glob.event1);
	if (err != CL_SUCCESS)return err;
	//printf("after 2\n");

	err  = clSetKernelArg(glob.vel_est_kernel,    0, sizeof(cl_mem), &glob.Z);
	err |= clSetKernelArg(glob.vel_est_kernel,    1, sizeof(cl_mem), &glob.temp_re);
	err |= clSetKernelArg(glob.vel_est_kernel,    2, sizeof(cl_mem), &glob.temp_im);
	err |= clSetKernelArg(glob.vel_est_kernel,    5, sizeof(cl_mem), &glob.std_dev);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.vel_est_kernel,    1, NULL, &glob.globWrkSize,            &glob.locWrkSize,            1, &glob.event1, &glob.event2);
	if (err != CL_SUCCESS)return err;
	//printf("after 3\n");

	err  = clSetKernelArg(glob.arctan_kernel,     0, sizeof(cl_mem),   &glob.temp_re);
	err |= clSetKernelArg(glob.arctan_kernel,     1, sizeof(cl_mem),   &glob.temp_im);
	err |= clSetKernelArg(glob.arctan_kernel,     2, sizeof(cl_float), &scale);                  // derived parameter
	err |= clSetKernelArg(glob.arctan_kernel,     3, sizeof(cl_int),   &glob.params.numb_avg);
	err |= clSetKernelArg(glob.arctan_kernel,     4, sizeof(cl_int),   &glob.params.avg_offset);
	err |= clSetKernelArg(glob.arctan_kernel,     5, sizeof(cl_mem),   &glob.outbufZ);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.arctan_kernel,     1, NULL, &glob.arctan_globWrkSize,     &glob.arctan_locWrkSize,     1, &glob.event2, &glob.event3);
	if (err != CL_SUCCESS)return err;
	//printf("after 4\n");

	err  = clSetKernelArg(glob.to_vel_est_kernel, 0, sizeof(cl_mem), &glob.L);
	err |= clSetKernelArg(glob.to_vel_est_kernel, 1, sizeof(cl_mem), &glob.R);
	err |= clSetKernelArg(glob.to_vel_est_kernel, 5, sizeof(cl_mem), &glob.to_vel_est_sum12_re_im);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.to_vel_est_kernel, 1, NULL, &glob.to_vel_est_globWrkSize, &glob.to_vel_est_locWrkSize, 1, &glob.event3, &glob.event4);
	if (err != CL_SUCCESS)return err;
	//printf("after 5\n");
	
	err  = clSetKernelArg(glob.to_arctan_kernel,  0, sizeof(cl_mem), &glob.to_vel_est_sum12_re_im);
	err |= clSetKernelArg(glob.to_arctan_kernel,  6, sizeof(cl_mem), &glob.outbufZX); //maybe don't care?
	err |= clSetKernelArg(glob.to_arctan_kernel,  7, sizeof(cl_mem), &glob.outbufX);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.to_arctan_kernel,  1, NULL, &glob.to_arctan_globWrkSize,  &glob.to_arctan_locWrkSize,  1, &glob.event4, &glob.event5);
	if (err != CL_SUCCESS)return err;
	//printf("after 6\n");

	// Set Arguments for maxabsval
	err  = clSetKernelArg(glob.maxabsval_kernel, 0, sizeof(cl_mem),      &glob.outbufZ);
	err |= clSetKernelArg(glob.maxabsval_kernel, 1, sizeof(cl_float)*64, NULL);
	err |= clSetKernelArg(glob.maxabsval_kernel, 2, sizeof(cl_int),      &Nsamples);
	err |= clSetKernelArg(glob.maxabsval_kernel, 3, sizeof(cl_mem),      &glob.result1);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.maxabsval_kernel,  1, NULL, &glob.maxabsval_globWrkSize,  &glob.maxabsval_locWrkSize,  1, &glob.event5, &glob.event6);
	if (err != CL_SUCCESS)return err;
	//printf("after 7\n");

	// Set Arguments for maxabsval
	err  = clSetKernelArg(glob.maxabsval_kernel, 0, sizeof(cl_mem),      &glob.outbufX);
	err |= clSetKernelArg(glob.maxabsval_kernel, 1, sizeof(cl_float)*64, NULL);
	err |= clSetKernelArg(glob.maxabsval_kernel, 2, sizeof(cl_int),      &Nsamples);
	err |= clSetKernelArg(glob.maxabsval_kernel, 3, sizeof(cl_mem),      &glob.result2);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.maxabsval_kernel,  1, NULL, &glob.maxabsval_globWrkSize,  &glob.maxabsval_locWrkSize,  1, &glob.event6, &glob.event7);
	if (err != CL_SUCCESS)return err;
	//printf("after 8\n");
	
	// TODO: check this
	err  = clSetKernelArg(glob.maxabsval2_kernel, 0, sizeof(cl_mem), &glob.result1);
	err |= clSetKernelArg(glob.maxabsval2_kernel, 1, sizeof(cl_mem), &glob.result2);
	err |= clSetKernelArg(glob.maxabsval2_kernel, 2, sizeof(cl_int), &threads);
	err |= clSetKernelArg(glob.maxabsval2_kernel, 3, sizeof(cl_mem), &glob.maximum);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.maxabsval2_kernel,    1, NULL, &glob.maxabsval2_globWrkSize,    &glob.maxabsval2_locWrkSize,    1, &glob.event7, &glob.event8);
	if (err != CL_SUCCESS)return err;
	//printf("after 9\n");

	// Combine kernel arguments
	err  = clSetKernelArg(glob.combine_kernel,   0, sizeof(cl_mem), &glob.outbufZ);
	err |= clSetKernelArg(glob.combine_kernel,   1, sizeof(cl_mem), &glob.outbufX);
	err |= clSetKernelArg(glob.combine_kernel,   2, sizeof(cl_mem), &glob.maximum);	// derived parameter
	err |= clSetKernelArg(glob.combine_kernel,   3, sizeof(cl_int), &Nsamples);		// derived parameter
	err |= clSetKernelArg(glob.combine_kernel,   4, sizeof(cl_mem), &outbuf[0]);
	err |= clSetKernelArg(glob.combine_kernel,   5, sizeof(cl_mem), &outbuf[1]);
	if (err != CL_SUCCESS)return err;
	err = clEnqueueNDRangeKernel(clqueue, glob.combine_kernel,    1, NULL, &glob.combine_globWrkSize,    &glob.combine_locWrkSize,    1, &glob.event8, outEv);
	if (err != CL_SUCCESS)return err;
	//printf("after 10\n");
	return 0;
}

/// <summary>Executes program on CPU 
/// Not needed in current implementation
/// </summary>
PLUGIN_API int  ProcessMemIO(void* inbuf[], size_t numin, void* outbuf[], size_t numout)
{
    return 0;
}
