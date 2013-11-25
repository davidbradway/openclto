from pyuspplugin import BuffSize, SampleFormat, UspPlugin
import pyopencl as cl
import ctypes as ct
import numpy as np


plugin_name = '../../Demo/plugins/plugin_a.dll'
print ('Working with ' + plugin_name + '\n')

plugin = UspPlugin(plugin_name)


info = plugin.GetPluginInfo()
fields = [a for a in dir(info) if (a[0] != '_')];

for field in fields:
    exec('print ( "info.' + field+' = {0}".format( info.' + field +'))' )
    
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
plugin.InitializeCL(ctx, '../../Demo/plugins/')
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

print ('\n')
print ('Successful invokation = {0}\n'.format(success))


