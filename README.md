# densor
A hardware-accelerated dense tensor (n-dimensional array) for Rust
with automatic memory management powered by [freqfs](https://github.com/haydnv/freqfs).

Hardware acceleration is provided by the [custos-math](https://github.com/elftausend/custos-math) library
which uses [custos](https://github.com/elftausend/custos)
to provide a standard interface to CPU, OpenCL, and CUDA devices.

For a general-purpose Rust interface for OpenCL, see the [ocl](https://github.com/cogciprocate/ocl)
project. For more information on OpenCL in general, see:
 - [A Gentle Introduction to OpenCL](https://freecontent.manning.com/wp-content/uploads/a-gentle-introduction-to-opencl.pdf) by Matthew Scarpino
 - [The OpenCL C Programming Language](https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_C.html) published by the Khronos Group
