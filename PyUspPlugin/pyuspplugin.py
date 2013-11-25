# -*- coding: utf-8 -*-
"""
This module implements a simple interface to UspPlugin DLLs

The basic class in the module is UspPlugin which implements an API to
UspPlugin DLL.

The API functions are:
    UspPlugin
       GetPluginInfo
       Initialize
       InitializeCL
       Cleanup
       SetParams
       SetInBufSize
       Prepare
       GetOutBufSize
       ProcessCLIO
       ProcessMemIO

To get more information type:
    >>> import pyuspplugin
    >>> help pyuspplugin.UspPlugin


Furthermore, the module defines a couple of help classes:
    PluginInfo - Structure describing the resources needed by a plugin
    SampleFormat - Types of samples handled by the UspPlugin DLLs
    BuffSize - Structure to define size of individual buffer

"""

import ctypes as ct
import pyopencl as cl
import numpy as np


#-----------------------------------------------------------------------------
class SampleFormat:
    """Values that describe the format of the samples.
    SAMPLE_FORMAT_xxxx, where xxxx is one of:
        UINT8, UINT16, UINT16X2, INT8, INT16, INT16X2, FLOAT32
        FLOAT32X2, INT32, INT32X2
    """
    uint8 = ct.c_int(0)
    uint16 = ct.c_int(1)
    uint16x2 = ct.c_int(2)
    int8 = ct.c_int(3)
    int16 = ct.c_int(4)
    int16x2 = ct.c_int(5)
    float32 = ct.c_int(6)
    float32x2 = ct.c_int(7)
    int32 = ct.c_int(8)
    int32x2 = ct.c_int(9)
   
#-----------------------------------------------------------------------------

#
# Sample format table will be used for conversion between BKM types, C-types 
# NumPy types. It contains also a textual description of the types.
# When the header is read, the SampleFormat is given as an enumerated value, 
# which can be looked-up from the SampleFormat enumeration (above)
#
SampleFormatTbl = [('uint8',     SampleFormat.uint8,     np.uint8,   ct.c_byte,   1),
                   ('uint16',    SampleFormat.uint16,    np.uint16,  ct.c_uint16, 1),
                   ('uint16x2',  SampleFormat.uint16x2,  np.uint16,  ct.c_uint16, 2),
                   ('int8',      SampleFormat.int8,      np.int8,    ct.c_int8,   1),
                   ('int16',     SampleFormat.int16,     np.int16,   ct.c_int16,  1),
                   ('int16x2',   SampleFormat.int16x2,   np.int16,   ct.c_int16,  2),
                   ('float32',   SampleFormat.float32,   np.float32, ct.c_float,  1),
                   ('float32x2', SampleFormat.float32x2, np.float32, ct.c_float,  2),
                   ('int32',     SampleFormat.int32,     np.int32,   ct.c_int32,  1),
                   ('int32x2',   SampleFormat.int32x2,   np.int32,   ct.c_int32,  2) 
              ]

SampleBytes = {SampleFormat.uint8.value: 1,
    SampleFormat.uint16.value:2,
    SampleFormat.uint16x2.value:4,
    SampleFormat.int8.value:1,
    SampleFormat.int16.value:2,
    SampleFormat.int16x2.value:4,
    SampleFormat.float32.value:4,
    SampleFormat.float32x2.value:8,
    SampleFormat.int32.value:4,
    SampleFormat.int32x2.value:8,
    }


# ----------------------------------------------------------------------------
class PluginInfo(ct.Structure):
    """ Structure returning information about a plugin.
        Fields are NumInBuffers, NumOutBuffers, UseOpenCL, InCLMem, OutCLMem
    """
    pass

PluginInfo._fields_ =\
    [('NumInBuffers', ct.c_int),     # Number of input buffers
     ('NumOutBuffers', ct.c_int),    # Number of output buffers
     ('UseOpenCL', ct.c_int),        # Does the module use open cl ?
     ('InCLMem', ct.c_int),          # Are inputs OpenCL memory objects  (1 - yes, 0 - no)
     ('OutCLMem', ct.c_int), ]       # Are inputs OpenCL memory objects  (1 - yes, 0 - no)

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
class BuffSize(ct.Structure):
    """ Structure describing the size of a buffer.

        Fields
        ------
        sampleType - Integer. See SampleFormat
        width, height, depth - Number of elements in every direction
        widthLen, heightLen, depthLen - Dimensions measured in bytes

    """
    pass

BuffSize._fields_ =\
    [('sampleType', ct.c_int),
     ('width',      ct.c_size_t),    # Number of samples along innermost dimension
     ('height',     ct.c_size_t),    # Number of samples along second dimension
     ('depth',      ct.c_size_t),    # Number of samples along third dimension
     ('widthLen',   ct.c_size_t),    # Length along first (innermost) dimension in bytes
     ('heightLen',  ct.c_size_t),    # Length along second dimension in bytes
     ('depthLen',   ct.c_size_t), ]  # Length along third dimension in bytes


def BuffSizeCreate(sampleType, width, height, depth):
    """ Creates a new BuffSize object with no zero padding.
    
    INPUTS
    ------
        sampleType - A value of type SampleFormat. Can be just a number
        width - Number of elements along first dimension
        height - Number of elements along second dimension
        depth - Number of elements along third dimensions

    """
    
    if ('value' in dir(sampleType)):
        sampleType = sampleType.value
    
    numSmpBytes = SampleBytes[sampleType];
    bufSize = BuffSize()
    bufSize.sampleType = sampleType
    bufSize.width = width
    bufSize.height = height
    bufSize.depth = depth
    bufSize.widthLen = numSmpBytes * bufSize.width
    bufSize.heightLen = bufSize.widthLen * bufSize.height
    bufSize.depthLen = bufSize.depth * bufSize.heightLen
    
    return bufSize
    

#------------------------------------------------------------------------------

class DbgOclMem(ct.Structure):
    pass

DbgOclMem._fields_=\
    [('name', ct.c_char_p),       # Name of the variable
     ('mem', ct.c_void_p),        # cl_mem - Pointer to open-cl mem object
     ('bufSize', BuffSize), ]     # Size of the buffer being debugged

class DbgMem(ct.Structure):
    pass;
    
DbgMem._fields_=\
    [('name', ct.c_char_p),       # Name of the variable
     ('ptr', ct.c_void_p),        # cl_mem - Pointer to open-cl mem object
     ('bufSize', BuffSize), ]     # Size of the buffer being debugged


#-----------------------------------------------------------------------------
class UspPlugin():
    """Class that handles plug-in modules.

    Usage pattern with OpenCL
    -------------------------

    .. code:: python

    import pyopencl as cl
    import numpy as np

    import pyusplugin as pu

    plugin = pu.UspPlugin('Path-to-my-dll-with-extension')
    info = plugin.GetPluginInfo()

    if (info.UseOpenCL == 0):
        print ('Example supports OpenCL only')
        raise NotImplementedError

    ctx = cl.create_some_context()
    cmd = cl.CommandQueue(ctx)

    inbuf = np.random.rand(1024).astype(np.float32)
    output = np.empty_like(inbuf)

    # Define the sizes of the buffers
    insize = pu.BuffSize()  # Size
    sf = SampleFormat()
    insize.sampleType = sf.SAMPLE_FORMAT_FLOAT32
    insize.width = 1024
    insize.height = insize.depth = 1

    insize.widthLen = data_size * ct.sizeof(ct.c_float)
    insize.depthLen = insize.heightLen = insize.widthLen * insize.height;

    #
    plugin.InitializeCL(ctx, 'path-to-dll. Maybe needed to load CL source');
    plugin.SetInBufSize(insize, 0)

    plugin.Prepare()
    outsize = plugin.GetOutBufSize( 0 )

    mf = cl.mem_flags

    inmem = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = inbuf )
    outmem = cl.Buffer(ctx, mf.WRITE_ONLY, 4096)


    plugin.ProcessCLIO([inmem], [outmem], cmd)

    cl.enqueue_copy(cmd, outbuf, outmem)  # Read output

    # Inspect results


    """
    def __init__(self, dllname):
        self.hDLL = ct.CDLL(dllname)

        # Create prototypes for the functions
        GetPluginInfoProto = ct.CFUNCTYPE(None, ct.POINTER(PluginInfo))
        InitializeProto = ct.CFUNCTYPE(ct.c_int, ct.c_char_p)
        InitializeCLProto = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_char_p)  # cl_context, cl_device, char* path
        InitializeProto = ct.CFUNCTYPE(ct.c_int, ct.c_char_p)
        CleanupProto = ct.CFUNCTYPE(ct.c_int)
        SetParamsProto = ct.CFUNCTYPE(ct.c_int, ct.POINTER(ct.c_float), ct.c_size_t, ct.POINTER(ct.c_int), ct.c_size_t)
        SetInBufSizeProto = ct.CFUNCTYPE(ct.c_int, ct.POINTER(BuffSize), ct.c_int)
        PrepareProto = ct.CFUNCTYPE(ct.c_int)
        GetOutBufSizeProto = ct.CFUNCTYPE(ct.c_int, ct.POINTER(BuffSize), ct.c_int)

        ProcessCLIOProto = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_void_p)
        ProcessMemIOProto = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t)
        GetDbgOclMemProto = ct.CFUNCTYPE(ct.POINTER(DbgOclMem), ct.POINTER(ct.c_uint32))
        GetDbgMemProto = ct.CFUNCTYPE(ct.POINTER(DbgMem), ct.POINTER(ct.c_uint32))
        
                
        #DbgOclMem* __cdecl GetDbgOclMem(uint32_t* arrayLen)
        self._GetPluginInfo = GetPluginInfoProto(("GetPluginInfo", self.hDLL))
        self._Initialize = InitializeProto(("Initialize", self.hDLL))
        self._InitializeCL = InitializeCLProto(("InitializeCL", self.hDLL))
        self._Cleanup = CleanupProto(("Cleanup", self.hDLL))
        self._SetParams = SetParamsProto(("SetParams", self.hDLL))
        self._SetInBufSize = SetInBufSizeProto(("SetInBufSize", self.hDLL))
        self._Prepare = PrepareProto(("Prepare", self.hDLL))
        self._GetOutBufSize = GetOutBufSizeProto(("GetOutBufSize", self.hDLL))
        self._ProcessCLIO = ProcessCLIOProto(("ProcessCLIO", self.hDLL))
        self._ProcessMemIO = ProcessMemIOProto(("ProcessMemIO", self.hDLL))
        
        # Debug interface
        self._GetDbgOclMem = GetDbgOclMemProto(("GetDbgOclMem", self.hDLL))
        self._GetDbgMem = GetDbgMemProto(("GetDbgMem", self.hDLL))
        
        
    def GetPluginInfo(self):
        """ Returns information about the DLL - Using OpenCL etc."""
        info = PluginInfo()
        self._GetPluginInfo(ct.byref(info))
        return info

    def Initialize(self, pathToDll):
        """ Initialize DLL. Set path to DLL. """
        res = self._Initialize(ct.c_char_p(pathToDll))
        return res

    def InitializeCL(self, context, pathToDLL):
        """ Initialize a DLL which uses OpenCL.
            context is a OpenCL context created using pyopencl
        """
        
        res = self._InitializeCL(context.obj_ptr,
                                 context.devices[0].obj_ptr,
                                 ct.c_char_p(pathToDLL))
        res = 0        
        return res

    def Cleanup(self):
        """ Releas allocated resources. Called before DLL is unloaded"""
        res = self._Cleanup()
        return res

    def SetParams(self, floatParams, intParams=[]):
        """Set parameters to the processing module.
        INPUTS
        ------
            floatParams - Array with floating point parameters
            intParams - Array with integer parameters

        OUTPUT
        ------
            0 if no error has occured, otherwise a value != 0
        """
        if (len(floatParams) > 0):
            fp = (ct.c_float * len(floatParams))()
            fp[:] = floatParams[:]
        else:
            fp = (ct.c_float)()

        if (len(intParams) > 0):
            ip = (ct.c_int * len(intParams))()
            ip[:] = floatParams[:]
        else:
            ip = (ct.c_int)()

        res = self._SetParams(fp, len(fp), ip, len(ip))
        return res

    def SetInBufSize(self, bufSize, bufnum=0):
        """ Set the size of input buffer.

        INPUTS
        ------
            bufSize: BuffSize() structure
                     Structure describing the size of an input buffer and the
                     type of the samples
            bufnum: Index of input buffer starting from 0

        OUTPUT
        ------
            0 if no errors
        """
        res = self._SetInBufSize(ct.byref(bufSize), bufnum)
        return res

    def Prepare(self):
        """ Prepare processing.

        Call this function BEFORE the actual processing

        INPUT
        -----
            None

        OUTPUT
        ------
            0 if no errors
        """
        res = self._Prepare()
        return res

    def GetOutBufSize(self, bufnum):
        """ Get the size of an output buffer

        USAGE
        -----
            size = obj.GetOutBufSize(bufnum)

        INPUT
        -----
            bufnum: Buffer index starting from 0

        OUTPUT
        ------
            size: BuffSize() structure
        """
        buf = BuffSize()
        res = self._GetOutBufSize(ct.byref(buf), bufnum)
        if (res != 0):
            print(' GetOutBufSize failed !')
        return buf

    def ProcessCLIO(self, inbufs, outbufs, cmdqueue, evin, evout):
        """ Process data where both inputs and output are OpenCL mem objects

        USAGE
        -----
            res = obj.ProcessCLIO(inbufs, outbufs, cmdqueue)

        INPUTS
        ------
            inbufs: list of pyopencl.Buffer() objects
            outbufs: list of pyopencl.Buffer() objects
            cmdqueue : pyopencl command queue
            evin : pyopencl user event - input
            evout : pyopencl user event - Filled in by the DLL

        REMARK
        ------
           If you have only 1 buffer, remember the []
            obj.Process([inbuf], [outbuf])
        OUTPUT
        ------
            0 if no errors

        """

        inbuf_array = (ct.c_void_p * len(inbufs))()  # Instantiate array of pointers
        for n in range(0, len(inbufs)):
            inbuf_array[n] = inbufs[n].obj_ptr

        outbuf_array = (ct.c_void_p * len(outbufs))()  # Instantiate array of pointers
        for n in range(0, len(outbufs)):
            outbuf_array[n] = outbufs[n].obj_ptr

        INTP = ct.POINTER(ct.c_int)
        ptr = INTP(ct.c_long(evout.obj_ptr))
        res = self._ProcessCLIO(inbuf_array, len(inbufs),
                                outbuf_array, len(outbufs),
                                cmdqueue.obj_ptr,
                                evin.obj_ptr,
                                ct.byref(ptr))
        return res

    def ProcessMemIO(self, inbufs, outbufs):
        """Process data with a module whose input/output buffers *ARE NOT* OpenCL buffers

        USAGE
        -----
            res = obj.ProcessCLIO(inbufs, outbufs, cmdqueue)

        INPUTS
        ------
            inbufs: list of numpy arrays
            outbufs: list of numpy arrays

        REMARK
        ------
           If you have only 1 buffer, remember the []
            obj.Process([inbuf], [outbuf])
        OUTPUT
        ------
            0 if no errors
        """
        inbuf_array = (ct.c_void_p * len(inbufs))()
        outbuf_array = (ct.c_void_p * len(outbufs))()

        for n in range(0, len(inbufs)):
            inbuf_array[n] = inbufs[n].ctypes.data_as(ct.c_void_p)

        for n in range(0, len(outbufs)):
            outbuf_array[n] = outbufs[n].ctypes.data_as(ct.c_void_p)

        res = self._ProcessMemIO(inbuf_array, len(inbufs), outbuf_array, len(outbufs))

        return res
        
    def DbgGetOclMem(self):
        """ Return a list of DbgOclMem structures and a list of OpenCL Buffers
        
        The memory definitions consist of BuffSize information and sampleFormat
        
        DbgOclMem structure consists of:
            name - Name of variable
            mem - Open CL mem object
            bufSize - Description of buffer (BuffSize())
            
        """
        
        numBufs = ct.c_uint()
        dbgOclMemPtr = self._GetDbgOclMem(ct.byref(numBufs))
        
        memDefs = []
        clBuffs = []
        
        for n in range(0, numBufs.value):
            B= DbgOclMem()
            B = dbgOclMemPtr[n]
            memDefs.append(B)                    
            clBuffs.append(cl.Buffer.from_cl_mem_as_int(B.mem))
            
        return (memDefs, clBuffs)
        
        
        
        
#---------- End of UspPlugin -------------------------------------------------


def main():
    plugin = UspPlugin(r'C:\Users\sin\Documents\Sources\plugins\distro_win32\bin\plugins\plugin_a.dll')
    # info = plugin.GetPluginInfo()
    data_size = 1024
    sf = SampleFormat()
    insize = BuffSize()
    outsize = BuffSize()

    insize.sampleType = sf.SAMPLE_FORMAT_FLOAT32
    insize.width = data_size
    insize.height = 1
    insize.depth = 1

    insize.widthLen = data_size * ct.sizeof(ct.c_float)
    insize.heightLen = insize.widthLen * insize.height
    insize.depthLen = insize.depth * insize.heightLen

    ctx = cl.create_some_context()
    plugin.InitializeCL(ctx, 'C:/Users/sin/Documents/Sources/plugins/distro_win32/bin/plugins/')
    plugin.SetInBufSize(insize, 0)

    plugin.Prepare()
    outsize = plugin.GetOutBufSize(0)

    # allocate buffers
    a = np.random.rand(data_size).astype(np.float32)

    mf = cl.mem_flags

    inbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    outbuf = cl.Buffer(ctx, mf.WRITE_ONLY, 4096)
    commands = cl.CommandQueue(ctx)
    event = cl.UserEvent(ctx)
    outEvent = cl.UserEvent(ctx)

    plugin.ProcessCLIO([inbuf], [outbuf], commands, event, outEvent)

    b = np.empty_like(a)
    cl.enqueue_copy(commands, b, outbuf)
    success = all(a ** 2 == b)

    print ('Successful invokation = {0}'.format(success))


if __name__ == '__main__':
    main()
