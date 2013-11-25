# -*- coding: utf-8 -*-
from pyuspplugin import BuffSize, BuffSizeCreate, SampleFormat, UspPlugin, DbgOclMem
import pyopencl as cl
import ctypes as ct
import numpy as np
import scipy.misc
from pylab import imshow, show

plug_path = r'C:/Users/sin/Documents/Sources/plugins/distro_vs_2012/bin/plugins/';

ctx = cl.create_some_context(interactive=False)

mod = UspPlugin(plug_path + 'plugin_b.dll');
err = mod.InitializeCL(ctx, plug_path)

Lena_32s = scipy.misc.lena()
Lena_32f = Lena_32s.astype(np.float32) * 111
Lena_8u = np.zeros(Lena_32f.shape, np.uint8)


sf = SampleFormat()
insize = BuffSize()
outsize = BuffSize()

insize = BuffSizeCreate(sf.float32, Lena_32f.shape[0], Lena_32f.shape[1], 1)
                        
mod.SetInBufSize(insize,0)
outsize = mod.GetOutBufSize(0)

mod.Prepare()
[dbgBufs, dbgMem] = mod.DbgGetOclMem()

mf = cl.mem_flags
inbuf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Lena_32f)
outbuf = cl.Buffer(ctx, mf.WRITE_ONLY, outsize.depthLen)


commands = cl.CommandQueue(ctx)
event = cl.UserEvent(ctx)
outEvent = cl.UserEvent(ctx)

mmin = np.zeros(512, dtype=np.float32)
#mmin[0] = Lena_32f.min()
mmax = np.zeros(512, dtype=np.float32)
#mmax[0] = Lena_32f.max()


cl.enqueue_write_buffer(commands, dbgMem[0], mmin)
cl.enqueue_write_buffer(commands, dbgMem[1], mmax)
commands.finish()

event.set_status(0)
mod.ProcessCLIO([inbuf], [outbuf], commands, event, outEvent)
commands.finish()

cl.enqueue_read_buffer(commands, outbuf, Lena_8u)
cl.enqueue_read_buffer(commands, dbgMem[0], mmin)
cl.enqueue_read_buffer(commands, dbgMem[1], mmax)


#imshow(Lena_8u), show()

