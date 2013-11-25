#pragma once

#include "USP/Modules/Module.h"
//#include "USP/Compute/OpenCL/ComputeOpenCL.h"
#include "UspPlugin.h"

#include <string>

TEST_CLASS_FORWARD_DECLARE(USPTests, UspPluginModuleTest)

namespace USP {

/// <summary> Module that loads an externally developed DLL which does some processing.
/// 		 The module is responsible for:
/// 		 <ul> 
///             <li> Load the DLL </li>
///             <li> Verify that all API functions are implemented </li>
///             <li> Query DLL's requirements - OpenCL or not </li>
///             <li> Verify sizes, count, and types of input/output buffers</li>
///             <li> Pass the parameters from the EngDataApi </li>
///             <li> Call the processing routine of the DLL </li>
///             <li> Call the cleanup() routine of the DLL and unload the DLL upon distruction</li>
/// 		 </ul>	   
/// 		 </summary>
class UspPluginModule : public Module {
public:

    // The following 4 functions are the "official interface o "
    UspPluginModule(Controller* controller);
    virtual ~UspPluginModule();
    virtual void InternalExecute(DataAdapter* inputDataAdapter);
    virtual void InternalCalc(IScanMan* scanMan);

	UspExtDllPluginParamType iParams; ///< Contains a copy of the parameters from the data model. Used in testing.


private:    
    void ClearApi();    ///< Set all pointers from the api structure to NULL
    void InitApi();     ///< Find the symbols from a loaded DLL and assign pointers to them
    void AllocBuffs();  ///< Allocate arrays of pointers to buffers passed to the loaded DLL
    void FreeBuffs();   ///< Free the allocated buffers
    std::shared_ptr<ComputeEvent> computeEvent;  ///< Used for synchronization

    std::vector<BuffSize> inBufSize;  
    std::vector<BuffSize> outBufSize;
    PluginApi api;       ///< Structure with pointers to functions implementing API
    PluginInfo info;     ///< The loaded DLL fills this structure and tells what it needs - OpenCL/CPU etc
    std::string dllName; ///< Full path to the DLL to be loaded. Not need be in System
    HMODULE hDLL;         ///< Handle to the DLL to be loaded
    
    // The processing modules take arrays of pointer to either memory or cl_mem
    (void**) inBufs;
    (void**) outBufs;

    cl_mem *inClMemPtr;
    cl_mem *outClMemPtr;

    TEST_CLASS_FRIEND_DECLARE(USPTests, UspPluginModuleTest)
};


}