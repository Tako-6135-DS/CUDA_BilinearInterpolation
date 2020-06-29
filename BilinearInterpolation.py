import pycuda.autoinit
from pycuda import driver, compiler, gpuarray
import numpy as np
import cv2
import timeit

def bil_px(image, w, h):
    scale = 2
    out = []
    i = int(w/scale)
    j = int(h/scale)
    k = 0 if i + 1 >= image.shape[0] else i + 1
    l = 0 if j + 1 >= image.shape[1] else j + 1
    a = image[i, j]/255
    b = image[k,j]/255
    c = image[i,l]/255
    d = image[k,l]/255
    result = (a + b + c + d)*255*0.25
    return result

def bilinear(image):
    preres = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.uint32)
    w = preres.shape[0]
    h = preres.shape[1]
    for i in range(w):
        for j in range(h):
            preres[i, j] = bil_px(image, i, j)
    return preres

IMG = 'rose.bmp'
image = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)
M1, N1 = image.shape
M2 = int(2*M1)
N2 = int(2*N1)

result = np.zeros((M2, N2), dtype=np.uint32)
block = (16, 16, 1)
grid = (int(np.ceil(M2/block[0])),int(np.ceil(N2/block[1])))

mod = compiler.SourceModule(open("kernel.cu", "r").read())
bilinear_interpolation_kernel = mod.get_function("interpolate")

x_out = np.array([i for i in range(M2)]*N2)
y_out = np.array([i for i in range(N2)]*M2)

start = driver.Event()
stop = driver.Event()

start.record()

tex = mod.get_texref("tex")
tex.set_filter_mode(driver.filter_mode.LINEAR)
tex.set_address_mode(0, driver.address_mode.CLAMP)
tex.set_address_mode(1, driver.address_mode.CLAMP)
driver.matrix_to_texref(image.astype(np.uint32), tex, order="C")

bilinear_interpolation_kernel(driver.Out(result), driver.In(x_out), driver.In(y_out), np.int32(M1), np.int32(N1), np.int32(M2), np.int32(N2), block=block, grid=grid, texrefs=[tex])
stop.record()
stop.synchronize()
gpu_time = stop.time_since(start)
print("GPU calculation time %.3f ms" % (gpu_time))

start = timeit.default_timer()
cpu_result = bilinear(image)
cpu_time = timeit.default_timer() - start
print("CPU calculation time %.3f ms" % (cpu_time))
