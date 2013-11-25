#define USP_PLUGIN_DLL   1
#include "UspPlugin.h"
#include "UspDebug.h"
#undef USP_PLUGIN_DLL 

#include <vector>

/* Global, module-wide variables */
static bool g_UspDebugInitialized = false;
static std::vector<DbgMem>  g_DbgMem; 
static std::vector<DbgOclMem> g_DbgOclMem; 

size_t numBytesPerSample[NUM_SAMPLE_FORMATS];


void InitializeUspDebug()
{
    numBytesPerSample[SAMPLE_FORMAT_UINT8]     = sizeof(uint8_t);
    numBytesPerSample[SAMPLE_FORMAT_UINT16]    = sizeof(uint16_t);
    numBytesPerSample[SAMPLE_FORMAT_UINT16X2]  = 2 * sizeof(uint16_t);
    numBytesPerSample[SAMPLE_FORMAT_INT8]      = sizeof(int8_t);
    numBytesPerSample[SAMPLE_FORMAT_INT16]     = sizeof(int16_t);
    numBytesPerSample[SAMPLE_FORMAT_INT16X2]   = 2 * sizeof(int16_t);
    numBytesPerSample[SAMPLE_FORMAT_FLOAT32]   = sizeof(float);
    numBytesPerSample[SAMPLE_FORMAT_FLOAT32X2] = 2 * sizeof(float);
    numBytesPerSample[SAMPLE_FORMAT_INT32]     = sizeof(int32_t);
    numBytesPerSample[SAMPLE_FORMAT_INT32X2]   = 2*sizeof(int32_t);

    g_DbgMem.clear();
    g_DbgOclMem.clear();

    g_UspDebugInitialized = true;
}


void DbgOclMemAppend(DbgOclMem dbgOclMem)
{
    if ( !g_UspDebugInitialized ) InitializeUspDebug();
    g_DbgOclMem.push_back(dbgOclMem);
}


void DbgMemAppend(DbgMem dbgMem)
{
    if ( !g_UspDebugInitialized ) InitializeUspDebug();
    g_DbgMem.push_back(dbgMem);
}


PLUGIN_API 
DbgOclMem*  GetDbgOclMem(uint32_t* arrayLen)
{
    if ( !g_UspDebugInitialized ) InitializeUspDebug();
    
    if (arrayLen != NULL){
        *arrayLen = g_DbgOclMem.size();
    }

    return g_DbgOclMem.data();
} 



PLUGIN_API 
DbgMem*  GetDbgMem(uint32_t* arrayLen)
{
    if ( !g_UspDebugInitialized ) InitializeUspDebug();

    if (arrayLen != NULL){
        *arrayLen = g_DbgMem.size();
    }
    return g_DbgMem.data();
}

