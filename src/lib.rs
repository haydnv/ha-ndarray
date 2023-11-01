use smallvec::SmallVec;

/// global device configuration

#[cfg(feature = "opencl")]
struct CLContext {}

enum HostContext {
    Heap,
    Stack,
}

trait PlatformInstance {}

#[cfg(feature = "opencl")]
pub struct OpenCL {}

#[cfg(feature = "opencl")]
impl PlatformInstance for OpenCL {}

pub struct Host;

impl PlatformInstance for Host {}

enum Platform {
    Host(Host),
    #[cfg(feature = "opencl")]
    CL(OpenCL),
}

impl PlatformInstance for Platform {}

/// buffers

#[cfg(feature = "opencl")]
pub trait CType: ocl::OclPrm {}

#[cfg(not(feature = "opencl"))]
pub trait CType {}

pub trait BufferInstance {}

impl<T: CType> BufferInstance for Vec<T> {}

impl<'a, T: CType> BufferInstance for &'a [T] {}

#[cfg(feature = "opencl")]
impl<T: CType> BufferInstance for ocl::Buffer<T> {}

/// arrays

pub type Shape = SmallVec<[usize; 8]>;

pub trait NDArray {}

pub struct ArrayBuffer<B> {
    buffer: B,
}

pub struct ArrayOp<Op> {
    op: Op,
}

pub struct ArraySlice<A> {
    source: A,
}

pub struct ArrayView<A> {
    source: A,
}

/// ops

pub trait Op {}

pub struct RandomUniform {}

impl Op for RandomUniform {}

pub struct Reduce<A> {
    array: A,
}

impl<A> Op for Reduce<A> {}

pub struct Add<L, R> {
    left: L,
    right: R,
}

impl<L, R> Op for Add<L, R> {}

pub struct MatMul<L, R> {
    left: L,
    right: R,
}

impl<L, R> Op for MatMul<L, R> {}
