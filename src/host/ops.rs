use std::f32::consts::PI;
use std::iter;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

use rand::Rng;
use rayon::join;
use rayon::prelude::*;

use crate::access::{Access, AccessBuffer};
use crate::ops::{Op, ReadValue, SliceSpec, ViewSpec};
use crate::{stackvec, strides_for, Axes, CType, Enqueue, Error, Float, Range, Shape};

use super::buffer::Buffer;
use super::platform::{Heap, Host, Stack};
use super::{SliceConverter, StackVec, VEC_MIN_SIZE};

pub struct CompareScalar<A, T> {
    access: A,
    scalar: T,
    cmp: fn(T, T) -> bool,
}

impl<A, T> CompareScalar<A, T> {
    pub fn eq(access: A, scalar: T) -> Self
    where
        T: PartialEq,
    {
        Self {
            access,
            scalar,
            cmp: |l, r| l == r,
        }
    }

    pub fn ge(access: A, scalar: T) -> Self
    where
        T: PartialOrd,
    {
        Self {
            access,
            scalar,
            cmp: |l, r| l >= r,
        }
    }

    pub fn gt(access: A, scalar: T) -> Self
    where
        T: PartialOrd,
    {
        Self {
            access,
            scalar,
            cmp: |l, r| l > r,
        }
    }

    pub fn le(access: A, scalar: T) -> Self
    where
        T: PartialOrd,
    {
        Self {
            access,
            scalar,
            cmp: |l, r| l <= r,
        }
    }

    pub fn lt(access: A, scalar: T) -> Self
    where
        T: PartialOrd,
    {
        Self {
            access,
            scalar,
            cmp: |l, r| l < r,
        }
    }

    pub fn ne(access: A, scalar: T) -> Self
    where
        T: PartialEq,
    {
        Self {
            access,
            scalar,
            cmp: |l, r| l != r,
        }
    }
}

impl<A: Access<T>, T: CType> Op for CompareScalar<A, T> {
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A: Access<T>, T: CType> Enqueue<Heap, u8> for CompareScalar<A, T> {
    type Buffer = Vec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .map(|l| if (self.cmp)(l, self.scalar) { 1 } else { 0 })
                    .collect()
            })
    }
}

impl<A: Access<T>, T: CType> Enqueue<Stack, u8> for CompareScalar<A, T> {
    type Buffer = StackVec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .as_ref()
                    .into_iter()
                    .copied()
                    .map(|l| if (self.cmp)(l, self.scalar) { 1 } else { 0 })
                    .collect()
            })
    }
}

impl<A: Access<T>, T: CType> Enqueue<Host, u8> for CompareScalar<A, T> {
    type Buffer = Buffer<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, u8>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, u8>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<A: Access<T>, T: CType> ReadValue<Host, u8> for CompareScalar<A, T> {
    fn read_value(&self, offset: usize) -> Result<u8, Error> {
        self.access
            .read_value(offset)
            .map(|n| if (self.cmp)(n, self.scalar) { 1 } else { 0 })
    }
}

pub struct Dual<L, R, IT, OT> {
    left: L,
    right: R,
    zip: fn(IT, IT) -> OT,
}

impl<L, R, IT, OT> Op for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Dual<L, R, T, T> {
    pub fn add(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Add::add,
        }
    }

    pub fn sub(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: Sub::sub,
        }
    }
}

impl<L, R, T: CType> Dual<L, R, T, u8> {
    pub fn eq(left: L, right: R) -> Self {
        Self {
            left,
            right,
            zip: |l, r| if l == r { 1 } else { 0 },
        }
    }
}

impl<L, R, IT, OT> Enqueue<Stack, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual(self.zip, left, right)
    }
}

impl<L, R, IT, OT> Enqueue<Heap, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual_parallel(self.zip, left, right)
    }
}

impl<L, R, IT, OT> Enqueue<Host, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, OT>::enqueue(self).map(Buffer::from)
        } else {
            Enqueue::<Heap, OT>::enqueue(self).map(Buffer::from)
        }
    }
}

impl<L, R, IT, OT> ReadValue<Host, OT> for Dual<L, R, IT, OT>
where
    L: Access<IT>,
    R: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        try_join_value(&self.left, &self.right, offset).map(|(l, r)| (self.zip)(l, r))
    }
}

pub struct Linear<T> {
    start: T,
    step: f64,
    size: usize,
}

impl<T> Linear<T> {
    pub fn new(start: T, step: f64, size: usize) -> Self {
        Self { start, step, size }
    }

    #[inline]
    fn value_at(&self, offset: usize) -> T
    where
        T: CType,
    {
        self.start + T::from_f64((offset as f64) * self.step)
    }
}

impl<T: Send + Sync> Op for Linear<T> {
    fn size(&self) -> usize {
        self.size
    }
}

impl<T: CType> Enqueue<Stack, T> for Linear<T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let start = self.start.to_float().to_f64();

        let buffer = (0..self.size)
            .into_iter()
            .map(|i| i as f64)
            .map(|i| i * self.step)
            .map(|o| start + o)
            .map(T::from_f64)
            .collect();

        Ok(buffer)
    }
}

impl<T: CType> Enqueue<Heap, T> for Linear<T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let buffer = (0..self.size)
            .into_par_iter()
            .map(|offset| self.value_at(offset))
            .collect();

        Ok(buffer)
    }
}

impl<T: CType> Enqueue<Host, T> for Linear<T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size < VEC_MIN_SIZE {
            Enqueue::<Stack, T>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, T>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<T: CType> ReadValue<Host, T> for Linear<T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        Ok(self.value_at(offset))
    }
}

pub struct MatMul<L, R, T> {
    left: L,
    right: R,
    batch_size: usize,
    dims: [usize; 3],
    dtype: PhantomData<T>,
}

impl<L, R, T> MatMul<L, R, T> {
    pub fn new(left: L, right: R, dims: [usize; 4]) -> Self {
        let [batch_size, a, b, c] = dims;

        Self {
            left,
            right,
            batch_size,
            dims: [a, b, c],
            dtype: PhantomData,
        }
    }
}

impl<L, R, T> Op for MatMul<L, R, T>
where
    L: Send + Sync,
    R: Send + Sync,
    T: Send + Sync,
{
    fn size(&self) -> usize {
        self.batch_size * self.dims[0] * self.dims[2]
    }
}

impl<L, R, T> Enqueue<Stack, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let left = self.left.read()?.to_slice()?;
        let right = self.right.read()?.to_slice()?;

        let [a, b, c] = self.dims;

        let mut product = StackVec::with_capacity(self.batch_size * a * c);

        for _batch in 0..self.batch_size {
            for x in 0..a {
                for z in 0..c {
                    let mut sum = T::ZERO;

                    for y in 0..b {
                        let l_offset = (x * b) + y;
                        let r_offset = (y * c) + z;
                        sum += left[l_offset] * right[r_offset];
                    }

                    product.push(sum)
                }
            }
        }

        debug_assert_eq!(product.len(), self.size());

        Ok(product)
    }
}

impl<L, R, T> Enqueue<Heap, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let [a, b, c] = self.dims;

        let (left, right) = try_join_read(&self.left, &self.right)?;

        // transpose the right matrices
        let right_size = b * c;
        let right_matrices = right.par_chunks_exact(right_size).map(|right| {
            let mut right_t = vec![T::ZERO; right_size];
            transpose::transpose(right, &mut right_t[..], c, b);
            right_t
        });

        let left_size = a * b;
        let left_matrices = left.par_chunks_exact(left_size);

        let output_size = a * c;
        let mut output = Vec::<T>::with_capacity(self.batch_size * output_size);
        let output_matrices = left_matrices
            .zip(right_matrices)
            .map(|(lm, rm)| {
                let mut out = Vec::<T>::with_capacity(output_size);

                let product = lm
                    .par_chunks_exact(b)
                    .map(|row| {
                        rm.par_chunks_exact(b).map(move |col| {
                            // chunk the dot product to encourage the compiler to vectorize
                            let col = col.par_chunks(8).map(|cc| cc.into_iter().copied());

                            row.par_chunks(8)
                                .zip(col)
                                .map(|(rc, cc)| {
                                    rc.into_iter().copied().zip(cc).map(|(r, c)| r * c).sum()
                                })
                                .sum::<T>()
                        })
                    })
                    .flatten();

                out.par_extend(product);
                out
            })
            .flatten();

        output.par_extend(output_matrices);

        debug_assert_eq!(output.len(), self.batch_size * output_size);

        Ok(output)
    }
}

impl<L, R, T> Enqueue<Host, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.left.size() < VEC_MIN_SIZE && self.right.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, T>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, T>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<L, R, T> ReadValue<Host, T> for MatMul<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, _offset: usize) -> Result<T, Error> {
        Err(Error::Bounds(
            "reading an individual value from a matrix multiplication is not implemented"
                .to_string(),
        ))
    }
}

pub struct RandomNormal {
    size: usize,
}

impl RandomNormal {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    fn box_muller(u: [f32; 2]) -> [f32; 2] {
        let [u1, u2] = u;
        let r = (u1.ln() * -2.).sqrt();
        let theta = 2. * PI * u2;
        [r * theta.cos(), r * theta.sin()]
    }
}

impl Op for RandomNormal {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<Heap, f32> for RandomNormal {
    type Buffer = Vec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut u = vec![
            0.0f32;
            if self.size % 2 == 0 {
                self.size
            } else {
                self.size + 1
            }
        ];

        rand::thread_rng().fill(&mut u[..]);

        let mut output = u
            .par_chunks_exact(2)
            .map(|u| {
                let u: [f32; 2] = u.try_into().expect("u");
                Self::box_muller(u)
            })
            .flatten()
            .collect::<Vec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }
}

impl Enqueue<Stack, f32> for RandomNormal {
    type Buffer = StackVec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut rng = rand::thread_rng();

        let mut output = iter::repeat_with(|| [rng.gen(), rng.gen()])
            .take(self.size.div_ceil(2))
            .map(Self::box_muller)
            .flatten()
            .collect::<StackVec<f32>>();

        if output.len() > self.size {
            output.pop();
        }

        debug_assert_eq!(output.len(), self.size);

        Ok(output)
    }
}

impl Enqueue<Host, f32> for RandomNormal {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size < VEC_MIN_SIZE {
            Enqueue::<Stack, f32>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, f32>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl ReadValue<Host, f32> for RandomNormal {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Err(Error::Bounds(
            "cannot calculate an individual value of a random normal distribution".to_string(),
        ))
    }
}

pub struct RandomUniform {
    size: usize,
}

impl RandomUniform {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Op for RandomUniform {
    fn size(&self) -> usize {
        self.size
    }
}

impl Enqueue<Heap, f32> for RandomUniform {
    type Buffer = Vec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut data = vec![0.; self.size];
        rand::thread_rng().fill(&mut data[..]);
        Ok(data)
    }
}

impl Enqueue<Stack, f32> for RandomUniform {
    type Buffer = StackVec<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let mut data = stackvec![0.; self.size];
        rand::thread_rng().fill(&mut data[..]);
        Ok(data)
    }
}

impl Enqueue<Host, f32> for RandomUniform {
    type Buffer = Buffer<f32>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size < VEC_MIN_SIZE {
            Enqueue::<Stack, f32>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, f32>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl ReadValue<Host, f32> for RandomUniform {
    fn read_value(&self, _offset: usize) -> Result<f32, Error> {
        Ok(rand::thread_rng().gen())
    }
}

pub struct Reduce<A, T> {
    access: A,
    stride: usize,
    reduce: fn(T, T) -> T,
    id: T,
}

impl<A, T> Reduce<A, T>
where
    T: CType,
{
    pub fn max(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: CType::max,
            id: T::MIN,
        }
    }

    pub fn min(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: CType::min,
            id: T::MAX,
        }
    }

    pub fn product(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: Mul::mul,
            id: T::ONE,
        }
    }

    pub fn sum(access: A, stride: usize) -> Self {
        Self {
            access,
            stride,
            reduce: Add::add,
            id: T::ZERO,
        }
    }
}

impl<A: Access<T>, T: CType> Op for Reduce<A, T> {
    fn size(&self) -> usize {
        debug_assert_eq!(self.access.size() % self.stride, 0);
        self.access.size() / self.stride
    }
}

impl<A: Access<T>, T: CType> Enqueue<Heap, T> for Reduce<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .chunks_exact(self.stride)
                    .map(|chunk| {
                        chunk
                            // encourage the compiler to vectorize
                            .par_chunks(8)
                            .map(|chunk| {
                                chunk.iter().copied().reduce(self.reduce).expect("reduced")
                            })
                            .reduce(|| self.id, self.reduce)
                    })
                    .collect()
            })
    }
}

impl<A: Access<T>, T: CType> Enqueue<Stack, T> for Reduce<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|slice| {
                slice
                    .chunks_exact(self.stride)
                    .map(|chunk| chunk.iter().copied().reduce(self.reduce).expect("reduced"))
                    .collect()
            })
    }
}

impl<A: Access<T>, T: CType> Enqueue<Host, T> for Reduce<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.stride < VEC_MIN_SIZE && self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, T>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, T>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<A: Access<T>, T: CType> ReadValue<Host, T> for Reduce<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let offset = offset * self.stride;

        if offset < self.access.size() {
            (offset..(offset + self.stride))
                .into_par_iter()
                .map(|offset| self.access.read_value(offset))
                .try_reduce(|| self.id, |r, v| Ok((self.reduce)(r, v)))
        } else {
            Err(Error::Bounds(format!(
                "invalid offset {offset} for a reduce op with size {}",
                self.size()
            )))
        }
    }
}

pub struct Slice<A, T> {
    access: A,
    spec: SliceSpec,
    dtype: PhantomData<T>,
}

impl<A, T> Slice<A, T> {
    pub fn new(access: A, shape: &[usize], range: Range) -> Self {
        let source_strides = strides_for(shape, shape.len()).collect();
        let spec = SliceSpec::new(range, source_strides);

        Self {
            access,
            spec,
            dtype: PhantomData,
        }
    }
}

impl<A: Send + Sync, T: Copy + Send + Sync> Slice<A, T> {
    fn read(&self, source: &[T]) -> Result<StackVec<T>, Error> {
        let output = (0..self.size())
            .into_iter()
            .map(|offset_out| self.spec.source_offset(offset_out))
            .map(|offset_in| source[offset_in])
            .collect();

        Ok(output)
    }

    fn read_parallel(&self, source: &[T]) -> Result<Vec<T>, Error> {
        let output = (0..self.size())
            .into_par_iter()
            .map(|offset_out| self.spec.source_offset(offset_out))
            .map(|offset_in| source[offset_in])
            .collect();

        Ok(output)
    }
}

impl<B, T> Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    fn overwrite(&mut self, data: &[T]) -> Result<(), Error> {
        if data.len() == self.size() {
            let source = self.access.as_mut().into_inner();

            for (offset, value) in data.into_iter().copied().enumerate() {
                let source_offset = self.spec.source_offset(offset);
                source.as_mut()[source_offset] = value;
            }

            Ok(())
        } else {
            Err(Error::Bounds(format!(
                "cannot overwrite a slice of size {} with a buffer of size {}",
                self.size(),
                data.len(),
            )))
        }
    }

    fn overwrite_value(&mut self, value: T) -> Result<(), Error> {
        let size = self.access.size();
        let source = self.access.as_mut().into_inner();

        for offset in 0..size {
            let source_offset = self.spec.source_offset(offset);
            source.as_mut()[source_offset] = value;
        }

        Ok(())
    }

    fn overwrite_value_at(&mut self, offset: usize, value: T) -> Result<(), Error> {
        let source_offset = self.spec.source_offset(offset);
        self.access.as_mut().into_inner().as_mut()[source_offset] = value;
        Ok(())
    }
}

impl<A: Send + Sync, T: Send + Sync> Op for Slice<A, T> {
    fn size(&self) -> usize {
        self.spec.size()
    }
}

impl<A: Access<T>, T: CType> Enqueue<Heap, T> for Slice<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .and_then(|buf| self.read_parallel(&*buf))
    }
}

impl<A: Access<T>, T: CType> Enqueue<Stack, T> for Slice<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .and_then(|buf| self.read(&*buf))
    }
}

impl<A: Access<T>, T: CType> Enqueue<Host, T> for Slice<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, T>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, T>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<A: Access<T>, T: CType> ReadValue<Host, T> for Slice<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        let offset = self.spec.source_offset(offset);
        self.access.read_value(offset)
    }
}

impl<'a, B, T> crate::ops::Write<'a, Heap, T> for Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.overwrite(data)
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.overwrite_value(value)
    }

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error> {
        self.overwrite_value_at(offset, value)
    }
}

impl<'a, B, T> crate::ops::Write<'a, Stack, T> for Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = &'a [T];

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.overwrite(data)
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.overwrite_value(value)
    }

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error> {
        self.overwrite_value_at(offset, value)
    }
}

impl<'a, B, T> crate::ops::Write<'a, Host, T> for Slice<AccessBuffer<B>, T>
where
    B: AsMut<[T]>,
    T: CType,
    AccessBuffer<B>: Access<T>,
{
    type Data = SliceConverter<'a, T>;

    fn write(&'a mut self, data: Self::Data) -> Result<(), Error> {
        self.overwrite(&*data)
    }

    fn write_value(&'a mut self, value: T) -> Result<(), Error> {
        self.overwrite_value(value)
    }

    fn write_value_at(&'a mut self, offset: usize, value: T) -> Result<(), Error> {
        self.overwrite_value_at(offset, value)
    }
}

pub struct Unary<A, IT, OT> {
    access: A,
    op: fn(IT) -> OT,
}

impl<A, T> Unary<A, T, T>
where
    A: Access<T>,
    T: CType,
{
    pub fn ln(access: A) -> Self {
        Self {
            access,
            op: |n| T::from_float(n.to_float().ln()),
        }
    }
}

impl<A, IT, OT> Op for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn size(&self) -> usize {
        self.access.size()
    }
}

impl<A, IT, OT> Enqueue<Heap, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Vec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|input| input.into_par_iter().copied().map(self.op).collect())
    }
}

impl<A, IT, OT> Enqueue<Stack, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = StackVec<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        self.access
            .read()
            .and_then(|buf| buf.to_slice())
            .map(|input| input.into_iter().copied().map(self.op).collect())
    }
}

impl<A, IT, OT> Enqueue<Host, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    type Buffer = Buffer<OT>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, OT>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, OT>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<A, IT, OT> ReadValue<Host, OT> for Unary<A, IT, OT>
where
    A: Access<IT>,
    IT: CType,
    OT: CType,
{
    fn read_value(&self, offset: usize) -> Result<OT, Error> {
        self.access.read_value(offset).map(|n| (self.op)(n))
    }
}

pub struct View<A, T> {
    access: A,
    spec: ViewSpec,
    dtype: PhantomData<T>,
}

impl<A: Access<T>, T: CType> View<A, T> {
    pub fn broadcast(access: A, shape: Shape, broadcast: Shape) -> Self {
        let strides = strides_for(&shape, broadcast.len()).collect();
        let source_strides = strides_for(&shape, shape.len()).collect();

        Self {
            access,
            spec: ViewSpec::new(broadcast, strides, source_strides),
            dtype: PhantomData,
        }
    }

    pub fn transpose(access: A, shape: Shape, axes: Axes) -> Self {
        let source_strides = strides_for(&shape, shape.len()).collect();
        let shape = axes.iter().copied().map(|x| shape[x]).collect::<Shape>();
        let strides = strides_for(&shape, shape.len()).collect();

        Self {
            access,
            spec: ViewSpec::new(shape, strides, source_strides),
            dtype: PhantomData,
        }
    }
}

impl<A: Access<T>, T: CType> Op for View<A, T> {
    fn size(&self) -> usize {
        self.spec.size()
    }
}

impl<A: Access<T>, T: CType> Enqueue<Stack, T> for View<A, T> {
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read().and_then(|source| source.to_slice())?;

        let buffer = (0..self.spec.size())
            .into_iter()
            .map(|offset| self.spec.source_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }
}

impl<A: Access<T>, T: CType> Enqueue<Heap, T> for View<A, T> {
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let source = self.access.read().and_then(|source| source.to_slice())?;

        let buffer = (0..self.spec.size())
            .into_par_iter()
            .map(|offset| self.spec.source_offset(offset))
            .map(|source_offset| source[source_offset])
            .collect();

        Ok(buffer)
    }
}

impl<A: Access<T>, T: CType> Enqueue<Host, T> for View<A, T> {
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, T>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, T>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<A: Access<T>, T: CType> ReadValue<Host, T> for View<A, T> {
    fn read_value(&self, offset: usize) -> Result<T, Error> {
        self.access.read_value(self.spec.source_offset(offset))
    }
}

fn exec_dual<IT: CType, OT: CType>(
    zip: fn(IT, IT) -> OT,
    left: SliceConverter<IT>,
    right: SliceConverter<IT>,
) -> Result<StackVec<OT>, Error> {
    let output = left
        .into_iter()
        .copied()
        .zip(right.into_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

fn exec_dual_parallel<IT: CType, OT: CType>(
    zip: fn(IT, IT) -> OT,
    left: SliceConverter<IT>,
    right: SliceConverter<IT>,
) -> Result<Vec<OT>, Error> {
    let output = left
        .into_par_iter()
        .copied()
        .zip(right.into_par_iter().copied())
        .map(|(l, r)| (zip)(l, r))
        .collect();

    Ok(output)
}

#[inline]
fn try_join_read<'a, L, R, T>(
    left: &'a L,
    right: &'a R,
) -> Result<(SliceConverter<'a, T>, SliceConverter<'a, T>), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    let (l, r) = join(
        || left.read().and_then(|buf| buf.to_slice()),
        || right.read().and_then(|buf| buf.to_slice()),
    );

    Ok((l?, r?))
}

#[inline]
fn try_join_value<'a, L, R, T>(left: &'a L, right: &'a R, offset: usize) -> Result<(T, T), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    let (l, r) = join(|| left.read_value(offset), || right.read_value(offset));

    Ok((l?, r?))
}
