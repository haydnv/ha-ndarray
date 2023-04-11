use std::marker::PhantomData;

use ocl::{Buffer, Error, Event, Kernel, OclPrm, Program, Queue};

use super::CDatatype;

pub trait Op<Out: OclPrm> {
    fn enqueue(&self, queue: Queue, output: Option<Buffer<Out>>) -> Result<Buffer<Out>, Error>;
}

// constructors

pub struct ArrayConstant<T> {
    value: T,
    size: u64,
}

impl<T> ArrayConstant<T> {
    pub fn new(value: T, size: u64) -> Self {
        Self { value, size }
    }
}

impl<T: OclPrm + CDatatype> Op<T> for ArrayConstant<T> {
    fn enqueue(&self, queue: Queue, output: Option<Buffer<T>>) -> Result<Buffer<T>, Error> {
        let src = format!(
            r#"
            __kernel void constant(
                __private {dtype} const value,
                __global float* const output)
            {{
                output[get_global_id(0)] = value;
            }}
        "#,
            dtype = T::TYPE_STR
        );

        let output = if let Some(output) = output {
            output
        } else {
            Buffer::builder()
                .queue(queue.clone())
                .len((self.size,))
                .build()?
        };

        let program = Program::builder()
            .source(src)
            .devices(queue.device())
            .build(&queue.context())?;

        let kernel = Kernel::builder()
            .program(&program)
            .name("constant")
            .queue(queue)
            .global_work_size((self.size,))
            .arg(self.value)
            .arg(&output)
            .build()?;

        let mut event = Event::empty();

        unsafe {
            kernel.cmd().enew(&mut event).enq()?;
        }

        Ok(output)
    }
}

pub struct ArrayRandom {
    size: u64,
}

pub struct MatEye {
    count: u64,
    size: u64,
}

// arithmetic

pub struct ArrayAdd<L, R> {
    left: L,
    right: R,
}

impl<L, R> ArrayAdd<L, R> {
    pub fn new(left: L, right: R) -> Self {
        Self { left, right }
    }
}

pub struct ArrayDiv<L, R> {
    left: L,
    right: R,
}

pub struct ArrayMul<L, R> {
    left: L,
    right: R,
}

pub struct ArrayMod<L, R> {
    left: L,
    right: R,
}

pub struct ArraySub<L, R> {
    left: L,
    right: R,
}

// linear algebra

pub struct MatDiag<A> {
    source: A,
}

pub struct MatMul<L, R> {
    left: L,
    right: R,
}

// comparison

pub struct ArrayEq<L, R> {
    left: L,
    right: R,
}

pub struct ArrayGT<L, R> {
    left: L,
    right: R,
}

pub struct ArrayGTE<L, R> {
    left: L,
    right: R,
}

pub struct ArrayLT<L, R> {
    left: L,
    right: R,
}

pub struct ArrayLTE<L, R> {
    left: L,
    right: R,
}

pub struct ArrayNE<L, R> {
    left: L,
    right: R,
}

// reduction

pub struct ArrayMax<A> {
    source: A,
}

pub struct ArrayMin<A> {
    source: A,
}

pub struct ArrayProduct<A> {
    source: A,
}

pub struct ArraySum<A> {
    source: A,
}

// other unary ops

pub struct ArrayCast<A, O> {
    source: A,
    dtype: PhantomData<O>,
}
