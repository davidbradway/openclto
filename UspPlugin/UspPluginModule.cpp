#include "StdAfx.h"

#include "UspPluginModule.h"
#include "EngConstants.h"
#include "IScanMan.h"

#include "USP/Util/util.h"
#include "USP/Controller.h"
#include "USP/Compute/OpenCL/ComputeOpenCL.h"
#include "USP/Compute/OpenCL/ComputeBufferOpenCL.h"
#include "USP/Compute/OpenCL/ComputeEventOpenCL.h"


#include "USP/DataStream.h"
#include "USP/DataAdapter.h"

#include "EngineUtils/MathUtils.h"
#include "EngineUtils/DataFormat.h"
#include "EngineUtils/Exception.h"


using namespace EngineUtils;
using namespace EngineUtils::StringUtils;

using namespace USP;

UspPluginModule::UspPluginModule(Controller* controller)
    : Module(controller, IMPLEMENTATION_TYPE_COMPUTE_GPU, 1)
{
    this->hDLL = 0L;
    this->dllName = "";
    this->inBufs = nullptr;
    this->inClMemPtr = nullptr;
    this->outBufs = nullptr;
    this->outClMemPtr = nullptr;

    this->computeEvent = GetCompute()->CreateComputeEvent();
}


UspPluginModule::~UspPluginModule()
{
    
    if (this->hDLL != NULL) {
		api.Cleanup();
        FreeLibrary(this->hDLL);
    }
    this->FreeBuffs();
}



void UspPluginModule::AllocBuffs()
{
    FreeBuffs();
    
    if (this->hDLL == NULL) return;  // Nothing to allocate
    
    if (this->info.InCLMem){
        this->inClMemPtr = new cl_mem [this->info.NumInBuffers];
        /* The values of cl_mem will be*/
    }else{
       this->inBufs = new void* [this->info.NumInBuffers];
       for (int n=0; n < this->info.NumInBuffers; n++)
       {
           this->inBufs[n] = _aligned_malloc(this->inBufSize[n].depthLen, 16);
           if (this->inBufs[n] == nullptr) {
               assert(false);
               throw EngineUtils::Exception("Could not allocate memory buffer");
           }
       }
    }

    if (this->info.OutCLMem){
        this->outClMemPtr = new cl_mem [this->info.NumOutBuffers];
        /* The values of cl_mem will be*/
    }else{
        this->outBufs = new void* [this->info.NumOutBuffers];
        for (int n=0; n < this->info.NumOutBuffers; n++)
        {
            this->outBufs[n] = _aligned_malloc(this->outBufSize[n].depthLen, 16);
            if (this->outBufs[n] == nullptr) {
                assert(false);
                throw EngineUtils::Exception("Could not allocate memory buffer");
            }
        }
    }
    
}



void UspPluginModule::FreeBuffs()
{
    if (this->inClMemPtr != nullptr) {
        delete [] this->inClMemPtr;
        this->inClMemPtr = nullptr;
    }

    if (this->outClMemPtr != nullptr) {
        delete [] this->outClMemPtr;
        this->outClMemPtr = nullptr;
    }

    if (this->inBufs != nullptr) {
        for (int n = 0; n < this->info.NumInBuffers; n++) {
            _aligned_free(this->inBufs[n]);
            this->inBufs[n] = nullptr;
        }
        delete [] this->inBufs;
        this->inBufs = nullptr;
    }

    if (this->outBufs != nullptr) {
        for (int n = 0; n < this->info.NumOutBuffers; n++) {
            _aligned_free(this->outBufs[n]);
            this->outBufs[n] = nullptr;
        }
        delete [] this->outBufs;
        this->outBufs = 0;
    }

}


/// <summary> Converts SampleFormat to SampleType
/// Basically these are the same. They are both defined as ENUMs.
/// The SampleType is in a separate header, which can be used outside of the
/// Engine/Console software, and as such risks to get out of SYNC.
/// This is what necessitates the conversion.
/// </summary>
/// <exception cref="EngineUtils::Exception">   Thrown when format is unknown. </exception>
/// <param name="format"> Enumerated type describing the format of the samples in a buffer. </param>
/// <returns> The plugin equivalent of format.</returns>
inline SampleType SampleFormatToSampleType(SampleFormatType format)
{
    SampleType smpType;

    switch (format)
    {
    case SampleFormatUInt8:          ///< Unsigned 8 bit integer
        smpType = SAMPLE_FORMAT_UINT8;
        break;
    case SampleFormatUInt16:
        smpType = SAMPLE_FORMAT_UINT16;
        break;
    case SampleFormatUInt16X2:       ///< 2 16 bit unsigned integers
        smpType = SAMPLE_FORMAT_UINT16X2;
        break;
    case SampleFormatInt8:
        smpType = SAMPLE_FORMAT_INT8;
        break;
    case SampleFormatInt16:
        smpType = SAMPLE_FORMAT_INT16;
        break;
    case SampleFormatInt16X2:
        smpType = SAMPLE_FORMAT_INT16X2;
        break;
    case SampleFormatFloat32:
        smpType = SAMPLE_FORMAT_FLOAT32;
        break;
    case SampleFormatFloat32X2:
        smpType = SAMPLE_FORMAT_FLOAT32X2;
        break;
    case SampleFormatInt32:
        smpType = SAMPLE_FORMAT_INT32;
        break;
    case SampleFormatInt32X2:
        smpType = SAMPLE_FORMAT_INT32X2;
        break;
    case SampleFormatUInt15:
        smpType = SAMPLE_FORMAT_UINT15;
        break;
    default:
        assert(false);
        throw EngineUtils::Exception(std::string(__FUNCTION__) + std::string(":\n Unknown sample format (defined in Engine)"));
    }
    return smpType;

}

/// <summary>  Complimentary function of SampleFormatToSampleType() </summary>
/// <exception cref="EngineUtils::Exception">   Thrown when smpType has an unsupported value. </exception>
/// <param name="smpType">  Enumerated type describing the data-type of the samples in a buffer. </param>
/// <returns> The engine equivalent of smpType </returns>
inline SampleFormatType SampleTypeToSampleFormat(SampleType smpType)
{
    SampleFormatType smpFormat;

    switch (smpType)
    {
    case SAMPLE_FORMAT_UINT8:          ///< Unsigned 8 bit integer
        smpFormat = SampleFormatUInt8;
        break;
    case SAMPLE_FORMAT_UINT16 :
        smpFormat = SampleFormatUInt16;
        break;
    case SAMPLE_FORMAT_UINT16X2 :       ///< 2 16 bit unsigned integers
        smpFormat = SampleFormatUInt16X2;
        break;
    case SAMPLE_FORMAT_INT8 :
        smpFormat = SampleFormatInt8;
        break;
    case SAMPLE_FORMAT_INT16 :
        smpFormat = SampleFormatInt16;
        break;
    case SAMPLE_FORMAT_INT16X2 :
        smpFormat = SampleFormatInt16X2;
        break;
    case SAMPLE_FORMAT_FLOAT32 :
        smpFormat = SampleFormatFloat32;
        break;
    case SAMPLE_FORMAT_FLOAT32X2 :
        smpFormat = SampleFormatFloat32X2;
        break;
    case SAMPLE_FORMAT_INT32 :
        smpFormat = SampleFormatInt32;
        break;
    case SAMPLE_FORMAT_INT32X2 :
        smpFormat = SampleFormatInt32X2;
        break;
    case SAMPLE_FORMAT_UINT15 :
        smpFormat = SampleFormatUInt15;
        break;

    default:
        assert(false);
        throw EngineUtils::Exception(std::string(__FUNCTION__) + std::string(":\n Unknown sample format (defined in Engine)"));
    }
    return smpFormat;

}



void UspPluginModule::ClearApi()
{
    api.GetPluginInfo = nullptr;   ///< Get information about the Plugin.
    api.InitializeCL = nullptr;    ///< Pass OpenCL context, device. Do initialization
    api.Initialize = nullptr;      ///< Initialization for plug-ins that do not use OpenCL
    api.Cleanup = nullptr;         ///< Release reseources
    api.SetParams = nullptr;       ///< Set parameters. An array of floats and ints
    api.SetInBufSize = nullptr;    ///< Specify the size of the input buffer
    api.Prepare = nullptr;         ///< Prepare for processing
    api.GetOutBufSize = nullptr;   ///< Get output buffer size 
    api.ProcessCLIO = nullptr;     ///< Do processing on OpenCL inputs/outputs
    api.ProcessMemIO = nullptr;    ///< Do processing on pure memory objects
}


void UspPluginModule::InitApi()
{
    if (this->hDLL == NULL) 
    {
        assert(false);
        throw EngineUtils::Exception("There is no handle to module. Load module first !");
    }

    api.GetPluginInfo = (GetPluginInfoPtr) GetProcAddress( hDLL, "GetPluginInfo");
    api.Initialize = (InitializePtr) GetProcAddress( hDLL, "Initialize");
    api.InitializeCL = (InitializeCLPtr) GetProcAddress(hDLL,"InitializeCL");
    api.SetParams = (SetParamsPtr) GetProcAddress( hDLL, "SetParams");
    api.SetInBufSize = (SetInBufSizePtr) GetProcAddress(hDLL, "SetInBufSize");
    api.Prepare = (PreparePtr) GetProcAddress(hDLL, "Prepare");
    api.GetOutBufSize = (GetOutBufSizePtr) GetProcAddress(hDLL, "GetOutBufSize");
    api.ProcessCLIO = (ProcessCLIOPtr) GetProcAddress(hDLL, "ProcessCLIO");
    api.ProcessMemIO = (ProcessMemIOPtr) GetProcAddress(hDLL, "ProcessMemIO");
    api.Cleanup = (CleanupPtr) GetProcAddress(hDLL, "Cleanup");


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
        this->ClearApi();   // All pointers to NULL !
        assert(false);
        throw EngineUtils::Exception(" One or more functions from the API were not found \n");
    }
}




void UspPluginModule::InternalCalc(IScanMan* scanMan)
{
   // Compute *ocl = GetCompute();
    ComputeOpenCL *ocl = static_cast<ComputeOpenCL*> (GetCompute());

    if (scanMan != nullptr) {
        // In case of unit testing, iParams are filled-in by the testing class
        CEngDataModel* dataModel = scanMan->GetEngDataModelPtr();
        this->iParams = dataModel->GetConstUSPParams(GetAlgorithmParamIndex()).extDllPlugin;
    }


    std::string newDllName = iParams.dllFilePath.path;
    int err = 0;

    /*
     *  If the name of the DLL has changed, then load and initialize the DLL
     */
    if (newDllName != this->dllName) {
        if (this->hDLL != 0L) {
            if ( this->api.Cleanup != nullptr) {
                this->api.Cleanup();
            }
            FreeLibrary(this->hDLL);
            this->hDLL = 0L;
        }

        this->ClearApi();
        this->FreeBuffs();
        
        this->dllName = newDllName;
        char  fullPath[512];

        GetFullPathName(this->dllName.c_str(), sizeof(fullPath), fullPath, NULL); 

        this->hDLL = LoadLibrary(fullPath);
        if (this->hDLL == NULL) {
            throw EngineUtils::Exception("Could not load " + this->dllName);
        }
        
        this->InitApi();
        api.GetPluginInfo(&this->info);
        if ( this->info.UseOpenCL ) {
            err = api.InitializeCL(ocl->GetOpenCLContext(), ocl->GetDeviceID(), PathSplit(this->dllName).c_str());
        } else {
            err = api.Initialize(PathSplit(this->dllName).c_str());
        }
        if (err) {
            assert(false);
            throw EngineUtils::Exception("Call to Initialize() func in DLL returned an error !");
        }
    }
    
    if ( this->info.NumInBuffers !=  (int)GetNumInputs() ) {
        assert(false);
       throw EngineUtils::Exception(std::string(__FUNCTION__) + std::string(":\n Number of input data adapters is different than specified" ));
    }

 

    inBufSize.clear();
    outBufSize.clear();


    if (err) {
        assert(false);
        throw EngineUtils::Exception("DLL Cleanup() returned error .");
    }


    if ( this->info.InCLMem != this->info.OutCLMem ) {
        assert( false );
        throw EngineUtils::Exception("Output buffers must be same type as input buffers - either OpenCL or Memory, but not mixed !");
    }


    for (int n = 0; n < this->info.NumInBuffers; n++) {
        BuffSize size;
        
        size.sampleType = SampleFormatToSampleType(GetInputDataAdapter(n)->GetDataFormat().GetSampleFormat());
        size.width = GetInputDataAdapter(n)->GetDataFormat().GetNumSamplesPerLine();
        size.height = GetInputDataAdapter(n)->GetDataFormat().GetNumLines();
        size.depth = GetInputDataAdapter(n)->GetDataFormat().GetNumPlanes();

        size.widthLen = GetInputDataAdapter(n)->GetDataFormat().GetLineSizeBytes();
        size.heightLen = GetInputDataAdapter(n)->GetDataFormat().GetPlaneSizeBytes();
        size.depthLen = GetInputDataAdapter(n)->GetDataFormat().GetFrameSizeBytes();
                
        this->inBufSize.push_back(size);
        err = api.SetInBufSize(&size, (int)n);

        if (err){
            assert(false);
            std::string msg;
            msg = std::string(__FUNCTION__) + std::string(":\n Error in SetInBufSize() ");
            msg += std::string(" for buffer number ") + ToString<int>(n);
            throw EngineUtils::Exception(msg);
        }
    }
    
    err = api.SetParams( (float*)&iParams.floatParams[0], (size_t)iParams.numFloatParams, 
                         (int*)&iParams.intParams[0], (size_t)iParams.numIntParams);

    if (err) {
        assert(false);
        throw EngineUtils::Exception("DLL SetParams() returned an error !");
    }

	
	err = api.Prepare();
    if (err) {
        assert(false);
        throw EngineUtils::Exception("DLL Prepare() returned an error !");
    }


    for (int n = 0; n < this->info.NumOutBuffers; n++) {
        BuffSize size;
        err = api.GetOutBufSize(&size, n);
        this->outBufSize.push_back(size);

        if (err) {
            assert (false);
            std::string msg;
            msg = std::string(__FUNCTION__) + std::string(":\n Error in GetOutBufSize() ");
            msg += std::string(" for buffer number ") + ToString<size_t>(n);
            throw EngineUtils::Exception(msg);
        }
    }
    

    SetNumOutputAdapters(this->info.NumOutBuffers);

    for (int n=0; n < this->info.NumOutBuffers; n++) {
        SampleFormatType smpFormat = SampleTypeToSampleFormat(outBufSize[n].sampleType);
        DataFormat outputDataFormat = DataFormat(HeaderFormatRaw, smpFormat, 
                                                 (int) outBufSize[n].width, 
                                                 (int) outBufSize[n].height, 
                                                 (int) outBufSize[n].depth);
        GetOutputDataAdapter(n)->ChangeDataFormat(outputDataFormat);
    }

    this->AllocBuffs();
}




void UspPluginModule::InternalExecute(DataAdapter* inputDataAdapter)
{
    UNREFERENCED_PARAMETER(inputDataAdapter);

    ComputeOpenCL *ocl = static_cast<ComputeOpenCL *>(GetCompute());
    ComputeEventOpenCL* computeEventOpenCL = static_cast<ComputeEventOpenCL*>(computeEvent.get());

    if (this->info.InCLMem) { 
        // Fill-in array with input buffers
        for ( int n = 0; n < this->info.NumInBuffers; n++ ) {
            ComputeBufferOpenCL *buf = (ComputeBufferOpenCL *) GetInputDataAdapter(n)->GetComputeBufferForRead(this->computeEvent).get();
            this->inClMemPtr[n] = buf->GetClMemObj();
        }

        
        for ( int n = 0; n < this->info.NumOutBuffers; n++ ) {
            ComputeBufferOpenCL *buf = (ComputeBufferOpenCL *) GetOutputDataAdapter(n)->GetComputeBufferForWrite().get();
            this->outClMemPtr[n] = buf->GetClMemObj();
        }
        cl_event exeEvent;  
        this->api.ProcessCLIO(this->inClMemPtr, this->info.NumInBuffers, this->outClMemPtr, this->info.NumOutBuffers, ocl->GetOpenCLQueue(),  computeEventOpenCL->GetCLEvent(), &exeEvent);

        // Change computeEvents internal event member to use the result event from the dll
        computeEventOpenCL->ReplaceCLEvent(exeEvent);

        GetOutputDataAdapter(0)->CompleteComputeBufferWrite(computeEvent);
		RegisterCompleteEvent(computeEvent);
    }else{
        // Copy all input streams to arrays in memory
        for ( int n = 0; n < this->info.NumInBuffers; n++ ) {
			 ocl->ReadFromBuffer(GetInputDataAdapter(n)->GetComputeBufferForRead(nullptr), 
				                 0, 
								 (uint) this->inBufSize[n].depthLen, 
                                 this->inBufs[n]);
        }
        this->api.ProcessMemIO(this->inBufs, this->info.NumInBuffers, this->outBufs, this->info.NumOutBuffers);

        for ( int n = 0; n < this->info.NumOutBuffers; n++ ) {
            ocl->WriteToBuffer(GetOutputDataAdapter(n)->GetComputeBufferForWrite(),
                               0, 
                               (uint) this->outBufSize[n].depthLen, 
                               this->outBufs[n]);
        }

        GetOutputDataAdapter(0)->CompleteComputeBufferWrite(computeEvent);
        RegisterCompleteEvent(computeEvent);
    }


}


