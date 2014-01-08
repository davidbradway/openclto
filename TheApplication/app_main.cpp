/// <summary> Autocorrelation processing </summary>
/* Host file using OpenCL plugin library */

#ifdef WIN32
//#include <windows.h>
//#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <dlfcn.h>
#endif

//#include "common_cl_srv.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <time.h>
#include "UspPlugin.h"
#include "Parameters.h"

//CREATE MEMORY SPACE
short  data[8*DATA_SIZE_IN]; //16-bit integer
unsigned char resultsZ[DATA_SIZE_OUT]; //8-bit integer
unsigned char resultsX[DATA_SIZE_OUT]; //8-bit integer

PluginApi api;
//char dllpath[4096];
PluginInfo pluginInfo;

#if defined ( WIN32 )
HMODULE hLib;
const char* dllname = "plugins/plugin_b.dll";
#elif defined (__APPLE__)
const char* dllname = "plugins/libplugin_a.dylib";
void * hLib;
#endif

#ifdef WIN32
int LoadDLL(const char *name)
{
    hLib = LoadLibrary(name);
    if (hLib == NULL) {
        printf("Could not load library \n");
        return -1;
    }
    api.GetPluginInfo = (GetPluginInfoPtr) GetProcAddress( hLib, "GetPluginInfo");
    api.Initialize = (InitializePtr) GetProcAddress(hLib, "Initialize");
    api.InitializeCL = (InitializeCLPtr) GetProcAddress(hLib,"InitializeCL");
    api.SetParams = (SetParamsPtr) GetProcAddress(hLib, "SetParams");
    api.SetInBufSize = (SetInBufSizePtr) GetProcAddress(hLib, "SetInBufSize");
    api.Prepare = (PreparePtr) GetProcAddress(hLib, "Prepare");
    api.GetOutBufSize = (GetOutBufSizePtr) GetProcAddress(hLib, "GetOutBufSize");
    api.ProcessCLIO = (ProcessCLIOPtr) GetProcAddress(hLib, "ProcessCLIO");
    api.ProcessMemIO = (ProcessMemIOPtr) GetProcAddress(hLib, "ProcessMemIO");
    api.Cleanup = (CleanupPtr) GetProcAddress(hLib, "Cleanup");

    if (   api.GetPluginInfo == NULL 
        || api.Initialize == NULL
        || api.InitializeCL == NULL
        || api.SetParams == NULL
        || api.SetInBufSize == NULL
        || api.Prepare == NULL
        || api.GetOutBufSize == NULL
        || api.ProcessCLIO == NULL
        || api.ProcessMemIO == NULL
        || api.Cleanup == NULL )
    { // If a pointer is equal to NULL
        printf(" One or more functions from the API were not found \n");
        return -1;
    }
    return 0;
}
#else
int LoadDLL(const char *name)
{
    hLib = dlopen(name, RTLD_LAZY);
    if (!hLib){
        fprintf(stderr, "%s\n", dlerror());
        return -1;
    }

    dlerror();    /* Clear any existing error */
    api.GetPluginInfo = (GetPluginInfoPtr) dlsym(hLib, "GetPluginInfo");
    api.Initialize = (InitializePtr) dlsym(hLib, "Initialize");
    api.InitializeCL = (InitializeCLPtr) dlsym(hLib,"InitializeCL");
    api.SetParams = (SetParamsPtr) dlsym(hLib, "SetParams");
    api.SetInBufSize = (SetInBufSizePtr) dlsym(hLib, "SetInBufSize");
    api.Prepare = (PreparePtr) dlsym(hLib, "Prepare");
    api.GetOutBufSize = (GetOutBufSizePtr) dlsym(hLib, "GetOutBufSize");
    api.ProcessCLIO = (ProcessCLIOPtr) dlsym(hLib, "ProcessCLIO");
    api.ProcessMemIO = (ProcessMemIOPtr) dlsym(hLib, "ProcessMemIO");
    api.Cleanup = (CleanupPtr) dlsym(hLib, "Cleanup");

    if (   api.GetPluginInfo == NULL
        || api.Initialize == NULL
        || api.InitializeCL == NULL
        || api.SetParams == NULL
        || api.SetInBufSize == NULL
        || api.Prepare == NULL
        || api.GetOutBufSize == NULL
        || api.ProcessCLIO == NULL
        || api.ProcessMemIO == NULL
        || api.Cleanup == NULL )
    { // If a pointer is equal to NULL
        printf(" One or more functions from the API were not found \n");
        return -1;
    }
    return 0;
}
#endif

/// <summary> Check for Error and print out error code detail </summary>
void checkError(int err, char *detail){
	if(err<0){
		printf("Error: %s \nError code: %d",detail,err);
		exit(1);
	}
}

/// <summary> New Load Data file, single file to single memory block </summary>
int load_data_file(short* data, const char *filename){
	// Load simulated data from file
	size_t datasize;
	FILE *ptr_myfile=fopen(filename,"rb");
	if (!ptr_myfile){ printf("Unable to open file!"); return -1; }
	fseek(ptr_myfile, 0, SEEK_END);
	datasize = ftell(ptr_myfile)/sizeof(short);
	rewind(ptr_myfile);
	size_t count1 = fread(data,sizeof(short),datasize,ptr_myfile);
	fclose(ptr_myfile);
	if(count1 != datasize){
		printf("Size mismatch!");
		printf("count1 = %d\n",count1);
		return -2; 
	}
	return 0;
}

/// <summary> New Save OpenCL result to one file</summary>
int save_data_file(unsigned char* outZ, unsigned char* outX, size_t estimates, const char *filename){
	size_t count1;
	FILE *ptr_myfile=fopen(filename,"wb");
	if (!ptr_myfile){ printf("Unable to open file!"); return -1; }
	count1 = fwrite(outZ, sizeof(unsigned char), estimates, ptr_myfile);
	count1 = fwrite(outX, sizeof(unsigned char), estimates, ptr_myfile);
	fclose(ptr_myfile);
	if(count1 != estimates){printf("Size mismatch!"); printf("count1 = %d, estimates = %d\n",count1,estimates); return -2; }
	return 0;
}

/// <summary> Main function of The Application
/// program to test the plugins DLL
/// </summary>
int main(int argc, char *argv[])
{
	int err;
    
	const int numin = 1;
	BuffSize insize[numin];
	const int numout = 2;
	BuffSize outsize[numout];

	// OpenCL memory needed
	// PCL_MEM buffer input and output arrays
	cl_mem inbuf[numin];
	cl_mem outbuf[numout];
	
    cl_device_id device_id = NULL;    // compute device id 
    cl_context context = NULL;        // compute context
    cl_command_queue commands = NULL; // compute command queue
	
	// Parameters for SetParams
	float floatParams[25];
	uint32_t numFloatParams; 
	int intParams[25]; 
	uint32_t numIntParams; 
	
#ifndef __APPLE__
    cl_platform_id platforms[2];
    cl_uint num_platforms;
#endif
	
#ifdef WIN32
    UNREFERENCED_PARAMETER(argc);
#endif

    if (LoadDLL(dllname) != 0) {
        printf("Something is wrong with DLL. Exitting \n");
        return EXIT_FAILURE;
    }

    int gpu = 1;
	
#if defined ( WIN32 )
	num_platforms = 2;

	// Step 01: Get platform information
    err = clGetPlatformIDs( 2, platforms, &num_platforms);
	checkError(err,"Failed to Get Platform IDs!");

	// Step 02: Get information about the device
    err = clGetDeviceIDs(platforms[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	checkError(err,"Failed to Get Device IDs!");
#elif defined( __APPLE__ )

	// Step 01/02: Get platform/device information
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
	checkError(err,"Failed to Get Device IDs!");
#endif

	/*
	// Parameter values in DPBs SARUS test dataset
	intParams[ind_emissions]     = 32; //meas.CFM.N_emis
	intParams[ind_nlines]        = 8;  //preview_image.no_lines/3
	intParams[ind_nlinesamples]  = 1024; //size(samples,1)
	intParams[ind_numb_avg]      = 1; // not sampled at 35 MHz... only 1024 samples for 6 cm... 8/meas.CFM.f0*sarus_sys.rcv_fs
	intParams[ind_avg_offset]    = 1;
	intParams[ind_lag_axial]     = 1;
	intParams[ind_lag_TO]        = 2;
	intParams[ind_lag_acq]       = 1;
	numIntParams                 = 8; //IntParamCount;
	
	floatParams[ind_fs]	      = 35000000;
	floatParams[ind_f0]       =  3000000;
	floatParams[ind_c]        =     1482;
	floatParams[ind_fprf]     =       98.6842;
	floatParams[ind_depth]    = static_cast<float>(0.035);   // par.sys.depth for analysis
	floatParams[ind_lambda_X] = static_cast<float>(0.0032); // transverse wavelength  par.TO.lambda_zx
	numFloatParams            = 6; //FloatParamCount;
	*/
	
	// Parameter values in JBOs ProFocus test dataset
	intParams[ind_emissions]     = 16; //meas.CFM.N_emis
	intParams[ind_nlines]        = 28;  // (4)transmits * 7
	intParams[ind_nlinesamples]  = 208; //size(samples,1)
	intParams[ind_numb_avg]      = 6; // not sampled at 35 MHz... only 1024 samples for 6 cm... 8/meas.CFM.f0*sarus_sys.rcv_fs
	intParams[ind_avg_offset]    = 1;
	intParams[ind_lag_axial]     = 1;
	intParams[ind_lag_TO]        = 2;
	intParams[ind_lag_acq]       = 1;
	intParams[ind_interleave]    = 16; // 4ZZLR * (4)transmits
	numIntParams                 = 9; //IntParamCount;
	
	floatParams[ind_fs]	      = 7500000;
	floatParams[ind_f0]       = 5000000;
	floatParams[ind_c]        =    1540;
	floatParams[ind_fprf]     =      3106;
	floatParams[ind_depth]    = static_cast<float>(0.02);   // par.sys.depth for analysis
	floatParams[ind_lambda_X] = static_cast<float>(0.0022); // transverse wavelength  par.TO.lambda_zx
	numFloatParams            = 6; //FloatParamCount;

	/*		
	// Parameter values in one file from MJPs test dataset
	intParams[ind_emissions]     = 32;
	intParams[ind_nlines]        = 75; // (3) * 25
	intParams[ind_nlinesamples]  = 1136;
	intParams[ind_numb_avg]      = 40;
	intParams[ind_avg_offset]    = 1;
	intParams[ind_lag_axial]     = 1;
	intParams[ind_lag_TO]        = 2;
	intParams[ind_lag_acq]       = 1;
	intParams[ind_interleave]    = 12; // 4 * (3)
	numIntParams                 = 9; //IntParamCount;
	
	floatParams[ind_fs]	      = 17500000;
	floatParams[ind_f0]       =  3500000;
	floatParams[ind_c]        =     1540;
	floatParams[ind_fprf]     =     2400;
	floatParams[ind_depth]    = static_cast<float>(0.03);   // 3cm focal depth
	floatParams[ind_lambda_X] = static_cast<float>(0.0033); // 3.3 mm transverse wavelength  lambda_zx: 0.0033
	numFloatParams            = 6; //FloatParamCount;
	*/

	// Step 03: Create OpenCL Context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	checkError(err,"Failed to create a compute context!");
	
    // Step 04: Create Command Queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
	checkError(err,"Failed to create a command queue!");

	// Step 05: Load the data from file
	const char  filename[] = "PF.bin";
	err = load_data_file(data,filename);
	checkError(err,"load data file failed");

	// Step 06: Read kernel file
	// PluginInfo tells us also if DLL uses OpenCL. We assume it does
    api.GetPluginInfo( &pluginInfo );   
	// Define path to kernel source code file
	char clKernelFilePath[] = ".\\plugins";
	
	// Step 07: Create Kernel program from the source
	err = api.InitializeCL(context, device_id, clKernelFilePath);
	checkError(err,"Failed initialization of CL");

    // Set the parameter arrays and numbers
	err |= api.SetParams(floatParams, numFloatParams, intParams, numIntParams);

	int i;
	for(i=0;i<numin;i++){
		// Size of input Buffer
		insize[i].sampleType = SAMPLE_FORMAT_INT16X2;
		insize[i].width      = intParams[ind_nlinesamples]*4*intParams[ind_nlines]*intParams[ind_emissions]; // 4 = 4CCLR
		insize[i].height     = 1;
		insize[i].depth      = 1;

		insize[i].widthLen  = insize[i].width  * sizeof(short)*2;
		insize[i].heightLen = insize[i].height * insize[i].widthLen;
		insize[i].depthLen  = insize[i].depth  * insize[i].heightLen;
		err |= api.SetInBufSize(&insize[i], i);
		checkError(err,"Failed to Set Input Buffer Size");
	}

	err |= api.Prepare();

	for(i=0;i<numout;i++){
		// Size of output Buffer
		err |= api.GetOutBufSize(&outsize[i], i);
		// allocate buffer in host
	}
	checkError(err,"Failed CL preparation");
	
	if (outsize[0].depthLen != insize[0].depthLen/intParams[ind_emissions]/8/sizeof(short)*sizeof(signed char)) { 
		printf("Output size is not what is expected !!!! \n"); exit(1); 
	}
 	
	// Step 05: Create memory buffer objects
    // Create the input and output arrays in device memory for our calculation
	inbuf[0]  = clCreateBuffer(context, CL_MEM_READ_ONLY, 8*DATA_SIZE_IN *sizeof(short),       NULL, &err); checkError(err,"Create buffer failed1");
	outbuf[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  DATA_SIZE_OUT*sizeof(unsigned char), NULL, &err); checkError(err,"Create buffer failed3");
	outbuf[1] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,  DATA_SIZE_OUT*sizeof(unsigned char), NULL, &err); checkError(err,"Create buffer failed4");

	// Step 05: Create user event objects
	cl_event evHost1 = clCreateUserEvent(context, NULL);    // TheApplication uses these events to enqueue operations
	cl_event evDLL   = clCreateUserEvent(context, NULL);    // This event is returned by the DLL, and is used as a "done" flag
	
	// Step 05: Enqueue writing to the memory buffer
	err = clEnqueueWriteBuffer(commands, inbuf[0], CL_FALSE, 0, 8*DATA_SIZE_IN*sizeof(short), data, 0, NULL, &evHost1); checkError(err,"Failed to write to source memory 1!");

	// Step 10: Set OpenCL kernel argument
	// Step 11: Execute OpenCL kernel in data parallel
	err = api.ProcessCLIO(inbuf, numin, outbuf, numout, commands, evHost1, &evDLL);
	checkError(err,"Failed process CL I/O");
	
	// Step 12: Read (Transfer result) from the memory buffer
	err = clEnqueueReadBuffer(commands, outbuf[0], CL_TRUE, 0, DATA_SIZE_OUT*sizeof(unsigned char), resultsX, 1, &evDLL, NULL); checkError(err,"Failed to read output array x!");
	err = clEnqueueReadBuffer(commands, outbuf[1], CL_TRUE, 0, DATA_SIZE_OUT*sizeof(unsigned char), resultsZ, 1, &evDLL, NULL); checkError(err,"Failed to read output array z!");
	
	// Step 13: Free objects
    api.Cleanup();

	//printf("Save the data to files!\n"); // Save the data to files!
	const char  fileresults[] =  "results.bin"; err = save_data_file(resultsZ,resultsX,DATA_SIZE_OUT, fileresults ); checkError(err,"save data file failed");

	// Step 13: Free objects
    err = clReleaseEvent(evHost1);checkError(err,"Failed release of Event1");
	err = clReleaseEvent(evDLL);checkError(err,"Failed release of Event2");
	err = clReleaseCommandQueue(commands);checkError(err,"Failed release of command queue");
	err = clReleaseContext(context);checkError(err,"Failed release of context");
	err = clReleaseMemObject(inbuf[0]); checkError(err,"Failed release of memory1");
	err = clReleaseMemObject(outbuf[0]); checkError(err,"Failed release of memory2");
	err = clReleaseMemObject(outbuf[1]); checkError(err,"Failed release of memory3");
	err = clReleaseEvent(evHost1); checkError(err,"Failed release of event1");
	err = clReleaseEvent(evDLL);   checkError(err,"Failed release of eventDLL");
	return 0;
}
