# ha-ndarray
A hardware-accelerated n-dimensional array for Rust
with automatic memory management powered by [freqfs](https://github.com/haydnv/freqfs).

Hardware acceleration is implemented using [OpenCL](https://www.khronos.org/opencl/)
via [ocl](https://github.com/cogciprocate/ocl), a general-purpose Rust interface for OpenCL.

For more information on OpenCL in general, see:
 - [A Gentle Introduction to OpenCL](https://freecontent.manning.com/wp-content/uploads/a-gentle-introduction-to-opencl.pdf) by Matthew Scarpino
 - [The OpenCL C Programming Language](https://registry.khronos.org/OpenCL/specs/2.2/html/OpenCL_C.html) published by the Khronos Group
 - [This excellent overview of OpenCL kernel programming & optimization](https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf)
