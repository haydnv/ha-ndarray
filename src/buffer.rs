use crate::{host, CType};

pub trait BufferInstance<T: CType>: Send + Sync + Sized {
    fn size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub enum Buffer<T: CType> {
    #[cfg(feature = "opencl")]
    CL(ocl::Buffer<T>),
    Host(host::Buffer<T>),
}

impl<T: CType> BufferInstance<T> for Buffer<T> {
    fn size(&self) -> usize {
        match self {
            #[cfg(feature = "opencl")]
            Self::CL(buf) => buf.size(),
            Self::Host(buf) => buf.size(),
        }
    }
}
