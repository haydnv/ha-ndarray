use std::ops::Add;

use ocl::{Buffer, OclPrm, ProQue, Queue};

use super::ops::*;
use super::{AxisBound, CDatatype, Error, NDArray, NDArrayReduce, Shape};

pub struct ArrayBase<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: OclPrm + CDatatype> ArrayBase<T> {
    pub fn concatenate(arrays: Vec<Array<T>>, axis: usize) -> Result<Self, Error> {
        todo!()
    }

    pub fn constant(value: T, shape: Shape) -> Result<Self, Error> {
        let size = shape.iter().product();
        Ok(Self {
            data: vec![value; size],
            shape,
        })
    }

    pub fn eye(shape: Shape) -> Result<Self, Error> {
        let ndim = shape.len();
        if shape.len() < 2 || shape[ndim - 1] != shape[ndim - 2] {
            Err(Error::Bounds(format!(
                "invalid shape for identity matrix: {:?}",
                shape
            )))
        } else {
            todo!()
        }
    }

    pub fn random(shape: Shape) -> Result<Self, Error> {
        todo!()
    }
}

impl<T: OclPrm> NDArray for ArrayBase<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<'a, T: OclPrm> Add for &'a ArrayBase<T> {
    type Output = ArrayOp<ArrayAdd<Self, Self>>;

    fn add(self, rhs: Self) -> Self::Output {
        let shape = broadcast_shape(self.shape(), rhs.shape()).expect("add");
        let op = ArrayAdd::new(self, rhs);
        ArrayOp { op, shape }
    }
}

impl<T: OclPrm + CDatatype> NDArrayReduce<T> for ArrayBase<T> {
    fn all(&self) -> Result<bool, Error> {
        let src = format!(
            r#"
        __kernel void reduce_all(
                __global uint* flag,
                __global {dtype}* input)
        {{
            if (input[get_global_id(0)] == 0) {{
                flag[0] = 1;
            }}
        }}
        "#,
            dtype = T::TYPE_STR
        );

        let pro_que = ProQue::builder().src(src).dims((self.size(),)).build()?;

        let input = pro_que.create_buffer()?;
        input.write(&self.data).enq()?;

        let mut result = vec![0u8];
        let flag: Buffer<u8> = pro_que.create_buffer()?;
        flag.write(&result).enq()?;

        let kernel = pro_que
            .kernel_builder("reduce_all")
            .arg(&flag)
            .arg(&input)
            .build()?;

        unsafe { kernel.enq()? }

        flag.read(&mut result).enq()?;

        Ok(result == [0])
    }

    fn all_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAll<Self>>, Error> {
        todo!()
    }

    fn any(&self) -> Result<bool, Error> {
        let src = format!(
            r#"
        __kernel void reduce_any(
                __global uint* flag,
                __global {dtype}* input)
        {{
            if (input[get_global_id(0)] != 0) {{
                flag[0] = 1;
            }}
        }}
        "#,
            dtype = T::TYPE_STR
        );

        let pro_que = ProQue::builder().src(src).dims((self.size(),)).build()?;

        let input = pro_que.create_buffer()?;
        input.write(&self.data).enq()?;

        let mut result = vec![0u8];
        let flag: Buffer<u8> = pro_que.create_buffer()?;
        flag.write(&result).enq()?;

        let kernel = pro_que
            .kernel_builder("reduce_any")
            .arg(&flag)
            .arg(&input)
            .build()?;

        unsafe { kernel.enq()? }

        flag.read(&mut result).enq()?;

        Ok(result == [1])
    }

    fn any_axis(&self, axis: usize) -> Result<ArrayOp<ArrayAny<Self>>, Error> {
        todo!()
    }

    fn max(&self) -> Result<T, Error> {
        todo!()
    }

    fn max_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMax<Self>>, Error> {
        todo!()
    }

    fn min(&self) -> Result<T, Error> {
        todo!()
    }

    fn min_axis(&self, axis: usize) -> Result<ArrayOp<ArrayMin<Self>>, Error> {
        todo!()
    }

    fn product(&self) -> Result<T, Error> {
        todo!()
    }

    fn product_axis(&self, axis: usize) -> Result<ArrayOp<ArrayProduct<Self>>, Error> {
        todo!()
    }

    fn sum(&self) -> Result<T, Error> {
        todo!()
    }

    fn sum_axis(&self, axis: usize) -> Result<ArrayOp<ArraySum<Self>>, Error> {
        todo!()
    }
}

pub struct ArrayOp<Op> {
    op: Op,
    shape: Shape,
}

impl<Op> NDArray for ArrayOp<Op> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub struct ArraySlice<A> {
    source: A,
    bounds: Vec<AxisBound>,
    shape: Shape,
}

impl<A> NDArray for ArraySlice<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub struct ArrayView<A> {
    source: A,
    strides: Vec<u64>,
    shape: Shape,
}

impl<A> NDArray for ArrayView<A> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub enum Array<T: OclPrm> {
    Base(ArrayBase<T>),
    Slice(ArraySlice<Box<Self>>),
    View(ArrayView<Box<Self>>),
    Op(ArrayOp<Box<dyn super::ops::Op<T>>>),
}

impl<T: OclPrm> NDArray for Array<T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Base(base) => base.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::View(view) => view.shape(),
            Self::Op(op) => op.shape(),
        }
    }
}

#[inline]
fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Shape, Error> {
    if left.len() < right.len() {
        return broadcast_shape(right, left);
    }

    let mut shape = Vec::with_capacity(left.len());
    let offset = left.len() - right.len();

    for x in 0..offset {
        shape[x] = left[x];
    }

    for x in 0..right.len() {
        if right[x] == 1 || right[x] == left[x + offset] {
            shape[x + offset] = left[x + offset];
        } else if left[x + offset] == 1 {
            shape[x + offset] = right[x];
        } else {
            return Err(Error::Bounds(format!(
                "cannot broadcast dimensions {} and {}",
                left[x + offset],
                right[x]
            )));
        }
    }

    Ok(shape)
}
