pub use ocl::{Buffer, Device, DeviceType, Error, OclPrm, Platform, Queue};

pub use array::*;
use ops::*;

mod array;
mod ops;

pub type Shape = Vec<u64>;

pub trait CDatatype {
    const TYPE_STR: &'static str;
}

impl CDatatype for f32 {
    const TYPE_STR: &'static str = "float";
}

pub trait NDArray: Sized {
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn size(&self) -> u64 {
        self.shape().iter().product()
    }

    fn shape(&self) -> &[u64];
}

pub trait ArrayRead<T: OclPrm>: NDArray {
    fn read(&self, queue: Queue, output: Option<Buffer<T>>) -> Result<Buffer<T>, Error>;
}

pub trait ArrayWrite<O>: NDArray {
    fn write(&self, other: &O) -> Result<(), Error>;
}

pub trait ArrayCast<O>: NDArray {
    fn cast(&self) -> Result<ArrayOp<ops::ArrayCast<Self, O>>, Error>;
}

pub trait ArrayCompare<T, O>: NDArray {
    fn eq(&self, other: &O) -> Result<ArrayOp<ArrayEq<Self, O>>, Error>;

    fn gt(&self, other: &O) -> Result<ArrayOp<ArrayGT<Self, O>>, Error>;

    fn gte(&self, other: &O) -> Result<ArrayOp<ArrayGTE<Self, O>>, Error>;

    fn lt(&self, other: &O) -> Result<ArrayOp<ArrayLT<Self, O>>, Error>;

    fn lte(&self, other: &O) -> Result<ArrayOp<ArrayLTE<Self, O>>, Error>;

    fn ne(&self, other: &O) -> Result<ArrayOp<ArrayNE<Self, O>>, Error>;
}

pub trait NDArrayReduce<T>: NDArray {
    fn max(&self, axis: usize) -> Result<T, Error>;

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMax<Self>>, Error>;

    fn min(&self, axis: usize) -> Result<T, Error>;

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMin<Self>>, Error>;

    fn product(&self, axis: usize) -> Result<T, Error>;

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayProduct<Self>>, Error>;

    fn sum(&self, axis: usize) -> Result<T, Error>;

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArraySum<Self>>, Error>;
}

pub trait NDArrayTransform<T>: NDArray {
    fn broadcast(shape: Shape) -> Result<ArrayView<Self>, Error>;

    fn transpose(axes: Option<Vec<usize>>) -> Result<ArrayView<Self>, Error>;

    fn reshape(shape: Shape) -> Result<ArrayView<Self>, Error>;

    fn slice(bounds: Vec<AxisBound>) -> Result<ArraySlice<Self>, Error>;
}

pub enum AxisBound {
    At(u64),
    In(u64, u64, u64),
    Of(Vec<u64>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_array() {
        let _array = ArrayBase::constant(1, vec![2, 3]);
    }
}
