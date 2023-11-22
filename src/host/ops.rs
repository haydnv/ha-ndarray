use std::f32::consts::PI;
use std::iter;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

use rand::Rng;
use rayon::join;
use rayon::prelude::*;

use crate::access::{Access, AccessBuffer};
use crate::buffer::BufferConverter;
use crate::ops::{Op, ReadValue, SliceSpec, ViewSpec};
use crate::{stackvec, strides_for, Axes, CType, Enqueue, Error, Float, Range, Shape};

use super::buffer::Buffer;
use super::platform::{Heap, Host, Stack};
use super::{SliceConverter, StackVec, VEC_MIN_SIZE};

pub struct Compare<L, R, T> {
    left: L,
    right: R,
    cmp: fn(T, T) -> u8,
}

impl<L, R, T> Op for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Compare<L, R, T> {
    pub fn eq(left: L, right: R) -> Self {
        Self {
            left,
            right,
            cmp: |l, r| if l == r { 1 } else { 0 },
        }
    }
}

impl<L, R, T> Enqueue<Stack, u8> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual(self.cmp, left, right)
    }
}

impl<L, R, T> Enqueue<Heap, u8> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual_parallel(self.cmp, left, right)
    }
}

impl<L, R, T> Enqueue<Host, u8> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<u8>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, u8>::enqueue(self).map(Buffer::Stack)
        } else {
            Enqueue::<Heap, u8>::enqueue(self).map(Buffer::Heap)
        }
    }
}

impl<L, R, T> ReadValue<Host, u8> for Compare<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<u8, Error> {
        try_join_value(&self.left, &self.right, offset).map(|(l, r)| (self.cmp)(l, r))
    }
}

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

pub struct Dual<L, R, T> {
    left: L,
    right: R,
    zip: fn(T, T) -> T,
}

impl<L, R, T> Op for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn size(&self) -> usize {
        self.left.size()
    }
}

impl<L, R, T: CType> Dual<L, R, T> {
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

impl<L, R, T> Enqueue<Stack, T> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = StackVec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual(self.zip, left, right)
    }
}

impl<L, R, T> Enqueue<Heap, T> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Vec<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        let (left, right) = try_join_read(&self.left, &self.right)?;
        exec_dual_parallel(self.zip, left, right)
    }
}

impl<L, R, T> Enqueue<Host, T> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    type Buffer = Buffer<T>;

    fn enqueue(&self) -> Result<Self::Buffer, Error> {
        if self.size() < VEC_MIN_SIZE {
            Enqueue::<Stack, T>::enqueue(self).map(Buffer::from)
        } else {
            Enqueue::<Heap, T>::enqueue(self).map(Buffer::from)
        }
    }
}

impl<L, R, T> ReadValue<Host, T> for Dual<L, R, T>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    fn read_value(&self, offset: usize) -> Result<T, Error> {
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
    left: BufferConverter<IT>,
    right: BufferConverter<IT>,
) -> Result<StackVec<OT>, Error> {
    let left = left.to_slice()?;
    let right = right.to_slice()?;

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
    left: BufferConverter<IT>,
    right: BufferConverter<IT>,
) -> Result<Vec<OT>, Error> {
    let left = left.to_slice()?;
    let right = right.to_slice()?;

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
) -> Result<(BufferConverter<'a, T>, BufferConverter<'a, T>), Error>
where
    L: Access<T>,
    R: Access<T>,
    T: CType,
{
    let (l, r) = join(|| left.read(), || right.read());

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
