#
# vector_addition.py
#
# Simple example of vector addition written in PyCUDA
#
# Luke Sheneman
# sheneman@uidaho.edu
# April 2024
#

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel function for vector addition
vector_addition_kernel = SourceModule("""
    __global__ void vector_addition(float *a, float *b, float *c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
""")

# Vector size
n = 1000000

# Host (CPU) input vectors
host_a = np.random.randn(n).astype(np.float32)
host_b = np.random.randn(n).astype(np.float32)

# Device (GPU) memory allocation
device_a = cuda.mem_alloc(host_a.nbytes)
device_b = cuda.mem_alloc(host_b.nbytes)
device_c = cuda.mem_alloc(host_a.nbytes)

# Copy input vectors from host to device
cuda.memcpy_htod(device_a, host_a)
cuda.memcpy_htod(device_b, host_b)

# Get the vector_addition function from the compiled CUDA kernel
vector_addition = vector_addition_kernel.get_function("vector_addition")

# Define block and grid dimensions
block_dim = (256, 1, 1)
grid_dim = ((n + block_dim[0] - 1) // block_dim[0], 1)

# Launch the kernel
while True:
    vector_addition(device_a, device_b, device_c, np.int32(n), block=block_dim, grid=grid_dim)

# Allocate memory for the result on the host
host_c = np.empty_like(host_a)

# Copy the result from device to host
cuda.memcpy_dtoh(host_c, device_c)

# Verify the result
assert np.allclose(host_c, host_a + host_b)

print("Vector addition completed successfully!")
