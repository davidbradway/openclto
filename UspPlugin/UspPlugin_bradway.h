#pragma once   
/**\file UspPlugin.h
 * Header file with forward declarations of functions that are exported by a 
 * DLL implementing a run-time plug-in to the USP framework in 2300
 *
 *  To create a DLL, simply include this file in your project and implement all
 *  of the exported functions.
 * 
 *  The DLL must link to an OpenCL library, regardless of whether OpenCL is
 *  being used in the processing or not. 
 *
 *  The structure PluginInfo is used to tell the host-program whether OpenCL 
 *  is needed by the DLL and if the input and output buffers are open-cl memory
 *  objects or not.
 *
 *  The size of each individual input buffer is described using a structure 
 *  of the type BuffSize.
 *
 *   The typical use of a DLL is as follows:
 *
 *    - Load DLL and link to the functions
 *    GetPluginInfo(& info)
 *    if (info.UseOpenCL != 0) {
 *       InitializeCL( cl_context, cl_device_id ,  path_to_dll );
 *    }else{
 *       Initialize( path_to_dll );
 *    }
 *    
 *    for (each of the input buffers) {
 *       BuffSize size;
 *       Fill-in size
 *       SetInBufSize(&size, buffer_index)
 *    }
 *
 *    SetParams(array_of_floats, len_array_f, array_of_ints, len_array_i)
 *    Prepare()  // Do initialization of memory, loops etc.
 *
 *    for (each of the output buffers) {
 *       BuffSize size;
 *       GetOutBufSize(&size, buffer_index)
 *       allocate buffer in host
 *    }
 *    
 *   (while-there-is-data-to-be-processed) 
 *   {
 *      if (info.InCLMem && info.OutCLMem) {
 *         ProcessCLIO(cl_mem_in[], numin, cl_mem_out[], numout, cl_command);
 *      }else{
 *        ProcessMemIO(void_ptr[], numin, void_ptr[], numout);
 *      }
 *   }
 *
 *   Cleanup()
 *
 *   - Unload DLL from memory
 */

#ifdef __cplusplus
#define EXTERNC    extern "C"
#else
#define EXTERNC
#endif

#if defined( WIN32 )
  #ifdef USP_PLUGIN_DLL
    #define PLUGIN_API EXTERNC __declspec(dllexport)
  #else
    #define PLUGIN_API EXTERNC __declspec(dllimport)
  #endif

#elif (defined( __GNUC__ ) && defined (__unix__) )|| defined (__APPLE__)
  #define PLUGIN_API EXTERNC __attribute__((visibility("default")))
#endif

#include <stdint.h>
#ifdef __APPLE__
  #include <OpenCL/OpenCL.h>
#else
  #include <CL/cl.h>
#endif

/// <summary>   Values that represent SampleType.
///  Must be in sync with SampleFormat used in the USP
/// </summary>
typedef enum SampleType
{
    SAMPLE_FORMAT_UINT8=0,
    SAMPLE_FORMAT_UINT16,
    SAMPLE_FORMAT_UINT16X2,
    SAMPLE_FORMAT_INT8,
    SAMPLE_FORMAT_INT16,
    SAMPLE_FORMAT_INT16X2,
    SAMPLE_FORMAT_FLOAT32,
    SAMPLE_FORMAT_FLOAT32X2,
    SAMPLE_FORMAT_INT32,
    SAMPLE_FORMAT_INT32X2,
    NUM_SAMPLE_FORMATS,
    SAMPLE_FORMAT_FORCE_32BIT = 0x8FFFFFFF
} SampleType;


/// <summary> Describes the capabilities of the plugin </summary>
typedef struct PluginInfo{
    int NumInBuffers;     ///< Number of input buffers (streams)
    int NumOutBuffers;    ///< Number of output buffers (streams)
    int UseOpenCL;        ///< Does the module use open cl ?
    int InCLMem;          ///< Are inputs OpenCL memory objects  (1 - yes, 0 - no)
    int OutCLMem;          ///< Are inputs OpenCL memory objects  (1 - yes, 0 - no)
} PluginInfo;


/// <summary> Structure that describes the size of a buffer </summary>
typedef struct BuffSize{
    SampleType sampleType;
    size_t width;    ///< Number of samples along innermost dimension
    size_t height;   ///< Number of samples along second dimension
    size_t depth;    ///< Number of samples along third dimension
    
    size_t widthLen;  ///< Length along first (innermost) dimension in bytes (includes eventual padding)
    size_t heightLen; ///< Length along second dimension in bytes (includes eventual padding)
    size_t depthLen;  ///< Length along third dimension in bytes (includes eventual padding)
} BuffSize;

/// <summary> Structure containing scanner parameters </summary>
typedef struct ParamStruct{
	//Scanner parameters
	int Nsamples; //Number of samples from one pulse
	int emissions; //Number of emissions in same direction
	int numb_avg; //Number of estimates to average over
	int avg_offset;
	int nlinesamples;
	int data_size_out;

	int lag_axial; // = 1; //
	int lag_TO; // = 2; //
	int lag_acq; // = 1; //bradway

	float fs; //The sampling freqency. [Hz]
	float f0; //The central frequency of the excitation. [Hz]
	float c; //The speed of sound. [m/s]
	float fprf; //Pulse repetition frequency [Hz]
	float pi;
	
	float depth; // = 0.03; // 3cm focal depth
	float lambda_X; //.0033 m
	float lambda_Y; //.0037 m
	
	float scale; //To store computed value. [m/s]
	float k_axial; //To store computed value. [m/s]
	float k_X; //To store computed value. [m/s/sample]
	float k_Y; //To store computed value. [m/s/sample]
} ParamStruct;


#ifdef USP_PLUGIN_DLL
/* Forward declaration of functions that must be exported by the DLL */
PLUGIN_API void  GetPluginInfo(PluginInfo* info);
PLUGIN_API int  InitializeCL( cl_context ctx, cl_device_id id, char* path_to_dll );
PLUGIN_API int  Initialize( char* path_to_dll );
PLUGIN_API int  Cleanup(void);
//PLUGIN_API int  SetParams(float* pfp, size_t nfp, int* pip, size_t nip);
PLUGIN_API int  SetParams(ParamStruct parameter);
PLUGIN_API int  SetInBufSize(BuffSize* buf, int bufnum);
PLUGIN_API int  Prepare(void);
PLUGIN_API int  GetOutBufSize(BuffSize* buf, int bufnum);
//bradway
PLUGIN_API int  ProcessCLIO(cl_mem* real_inbufZ,  cl_mem* imag_inbufZ,
							cl_mem* real_inbufLX, cl_mem* imag_inbufLX,
							cl_mem* real_inbufRX, cl_mem* imag_inbufRX,
							cl_mem* real_inbufLY, cl_mem* imag_inbufLY, 
							cl_mem* real_inbufRY, cl_mem* imag_inbufRY, size_t numin, 
							cl_mem* outbufZ, cl_mem* outbufZX, cl_mem* outbufXX, cl_mem* outbufZY, cl_mem* outbufYY, size_t numout, 
							cl_command_queue  clqueue, cl_event inEv, cl_event* outEv);
PLUGIN_API int  ProcessMemIO(void* inbuf[], size_t numin, void* outbuf[], size_t numout);
#endif

typedef  void  (*GetPluginInfoPtr)(PluginInfo* info);
typedef  int  (*InitializeCLPtr)( cl_context ctx, cl_device_id id, const char* path_to_dll );
typedef  int  (*InitializePtr)( const char* path_to_dll );
typedef  int  (*CleanupPtr)(void);
//typedef  int  (*SetParamsPtr)(float* pfp, size_t nfp, int* pip, size_t nip);
typedef  int  (*SetParamsPtr)(ParamStruct parameter);
typedef  int  (*SetInBufSizePtr)(BuffSize* buf, int bufnum);
typedef  int  (*PreparePtr)(void);
typedef  int  (*GetOutBufSizePtr)(BuffSize* buf, int bufnum);
//bradway
typedef  int  (*ProcessCLIOPtr)(cl_mem* real_inbufZ,  cl_mem* imag_inbufZ,
								cl_mem* real_inbufLX, cl_mem* imag_inbufLX,
								cl_mem* real_inbufRX, cl_mem* imag_inbufRX,
								cl_mem* real_inbufLY, cl_mem* imag_inbufLY,
								cl_mem* real_inbufRY, cl_mem* imag_inbufRY, size_t numin, 
								cl_mem* outbufZ, cl_mem* outbufZX, cl_mem* outbufXX, cl_mem* outbufZY, cl_mem* outbufYY, size_t numout, 
								cl_command_queue  clqueue, cl_event inEv, cl_event* outEv);
typedef  int  (*ProcessMemIOPtr)(void* inbuf[], size_t numin, void* outbuf[], size_t numout);

/// <summary>  Structure that encapsulates the API. </summary>
typedef struct PluginApi
{
    GetPluginInfoPtr GetPluginInfo; ///< Get information about the Plugin.
    InitializeCLPtr InitializeCL;   ///< Pass OpenCL context, device. Do initialization
    InitializePtr Initialize;       ///< Initialization for plug-ins that do not use OpenCL
    CleanupPtr Cleanup;             ///< Release reseources
    SetParamsPtr SetParams;         ///< Set parameters. An array of floats and ints
    SetInBufSizePtr SetInBufSize;   ///< Specify the size of the input buffer
    PreparePtr Prepare;             ///< Prepare for processing
    GetOutBufSizePtr GetOutBufSize;  ///< Get output buffer size 
    ProcessCLIOPtr ProcessCLIO;      ///< Do processing on OpenCL inputs/outputs
    ProcessMemIOPtr ProcessMemIO;    ///< Do processing on pure memory objects
} PluginApi;
