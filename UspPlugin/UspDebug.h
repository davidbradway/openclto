#pragma once
/**
 * The module maintains two internal lists of buffer descriptions:
 * one for OpenCL buffers and one for memory buffers.
 * The lists are implemented as std::vector. 
 * 
 * The host application can query the DLL about these two lists,
 * but it must not free() them.
 * The external API consists of  GetOCLMems() and GetMems().
 * Example (for the host application):
 *
 *  #include <UspDebug.h>
 *  #include <stdio.h>
 *
 *  uint32_t numBufs; 
 *  DbgMem * dbgMem = GetMems(&numBufs); 
 * 
 *  for (uint32_t n = 0; n < numBufs; n++){
 *      printf("dbgMem[%d].name = %s", n, dbgMem[n].name);
 *  }
 * 
 * These two functions return pointers to memory
 * One can use the macros 'DBG_OCL_BUF?', 'DBG_MEM_BUF?' 
 * as a short cut for describing buffers that do not have 
 * zero-padding in any dimension. The (?) can be 1, 2 or 3 
 * and stands for the number of dimensions
 *
 */


#include "UspPlugin.h"


/** Debug information for a buffer allocated in memory */
typedef struct DbgOclMem {
    char *name;
    cl_mem mem;
    BuffSize bufSize;
} DbgOclMem;



/** Debug information for a buffer allocated in memory */
typedef struct DbgMem {
    char* name;  
    void* ptr;
    BuffSize bufSize;
} DbgMem;



#ifdef USP_PLUGIN_DLL

/**
 *  External API
 */

PLUGIN_API DbgOclMem*  GetDbgOclMem(uint32_t* arrayLen); 
PLUGIN_API DbgMem*  GetDbgMem(uint32_t* arrayLen);


/** Internal API
 * One can use the macros 'DBG_OCL_BUF?', 'DBG_MEM_BUF?' 
 * as a short cut for describing buffers that do not have 
 * zero-padding in any dimension. The (?) can be 1, 2 or 3 
 * and stands for the number of dimensions
 */
void DbgOclMemAppend(DbgOclMem dbgOclMem);
void DbgMemAppend(DbgMem dbgMem);

/** Macro definitions for appending 1, 2 and 3D OpenCL 
 * and Mem buffers to the debug list 
 */
#define DBG_OCL1(mem, smpType, dim0) {\
    DbgOclMem oclBufDescr = DBG_OCL_BUF1(mem, smpType, dim0);\
    DbgOclMemAppend(oclBufDescr);\
}

#define DBG_OCL2(mem, smpType, dim0, dim1) {\
    DbgOclMem oclBufDescr = DBG_OCL_BUF2(mem, smpType, dim0, dim1);\
    DbgOclMemAppend(oclBufDescr);\
}

#define DBG_OCL3(mem, smpType, dim0, dim1, dim2) {\
    DbgOclMem oclBufDescr = DBG_OCL_BUF3(mem, smpType, dim0, dim1, dim2);\
    DbgOclMemAppend(oclBufDescr);\
}


#define DBG_MEM1(mem, smpType, dim0) {\
    DbgMem memBufDescr = DBG_MEM_BUF1(mem, smpType, dim0);\
    DbgMemAppend(memBufDescr);\
}

#define DBG_MEM2(mem, smpType, dim0, dim1) {\
    DbgMem memBufDescr = DBG_MEM_BUF2(mem, smpType, dim0, dim1);\
    DbgMemAppend(memBufDescr);\
}

#define DBG_MEM3(mem, smpType, dim0, dim1, dim2) {\
    DbgMem memBufDescr = DBG_MEM_BUF3(mem, smpType, dim0, dim1, dim2);\
    DbgMemAppend(memBufDescr);\
}



/** Table defined in UspDebug.cpp */
extern size_t numBytesPerSample[];



/** Description of 1D OpenCL buffers */
#define DBG_OCL_BUF1(mem, smpType, dim0) {\
(char*)#mem,mem,\
{smpType,dim0,1,1,dim0*numBytesPerSample[smpType],\
1*dim0*numBytesPerSample[smpType],\
1*1*dim0*numBytesPerSample[smpType]}}

/** Description of 2D OpenCL buffers */
#define DBG_OCL_BUF2(mem, smpType, dim0, dim1) {\
(char*)#mem,mem,\
{smpType,dim0,dim1,1,dim0*numBytesPerSample[smpType],\
dim1*dim0*numBytesPerSample[smpType],\
1*dim1*dim0*numBytesPerSample[smpType]}}

/** Description of 3D OpenCL buffers */
#define DBG_OCL_BUF3(mem, smpType, dim0, dim1, dim2) {\
(char*)#mem,mem,\
{smpType,dim0,dim1,dim2,dim0*numBytesPerSample[smpType],\
dim1*dim0*numBytesPerSample[smpType],\
dim2*dim1*dim0*numBytesPerSample[smpType]}}





/** Description of 1D memory buffers */
#define DBG_MEM_BUF1(mem, smpType, dim0) {\
(char*)#mem,(void*)mem,\
{smpType,dim0,1,1,dim0*numBytesPerSample[smpType],\
1*dim0*numBytesPerSample[smpType],\
1*1*dim0*numBytesPerSample[smpType]}}

/** Description of 2D memory buffers */
#define DBG_MEM_BUF2(mem, smpType, dim0, dim1) {\
(char*)#mem,(void*)mem,\
{smpType,dim0,dim1,1,dim0*numBytesPerSample[smpType],\
dim1*dim0*numBytesPerSample[smpType],\
1*dim1*dim0*numBytesPerSample[smpType]}}

/** Description of 3D memory buffers */
#define DBG_MEM_BUF3(mem, smpType, dim0, dim1, dim2) {\
(char*)#mem,(void*)mem,\
{smpType,dim0,dim1,dim2,dim0*numBytesPerSample[smpType],\
dim1*dim0*numBytesPerSample[smpType],\
dim2*dim1*dim0*numBytesPerSample[smpType]}}


#endif

/** Type definitions used by the caller to bind to the API from the DLL */

typedef DbgOclMem* (*GetDbgOclMemPtr)(uint32_t* arrayLen); 
typedef DbgMem*    (*GetDbgMemPtr)(uint32_t* arrayLen);

/** Encapsulate the API in a structure */
typedef struct PluginDbgApi
{
    GetDbgOclMemPtr GetDbgOclMem;    //< Get array with pointers to OCL buffers
    GetDbgMemPtr GetDbgMem;          //< Get array with pointers to mem buffers
}PluginDbgApi;

