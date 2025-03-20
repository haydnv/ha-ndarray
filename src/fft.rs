use std::iter;

use num_complex::Complex;

use crate::{
    Access, Array, ArrayAccess, Axes, AxisRange, Error, NDArray, NDArrayFourier, NDArrayTransform,
    Number, Range,
};

/// Fast Fourier Transform, an alias of [`NDArrayFourier::fft`]
pub fn fft<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    data.fft()
}

/// Inverse Fast Fourier Transform, an alias of [`NDArrayFourier::ifft`]
pub fn ifft<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    data.ifft()
}

/// Two-dimensional Fast Fourier Transform
pub fn fft2<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    if data.ndim() >= 2 {
        let mut permutation = (0..data.ndim()).into_iter().collect::<Axes>();
        permutation.swap(data.ndim() - 1, data.ndim() - 2);

        data.fft()?
            .transpose(permutation.clone())?
            .fft()?
            .transpose(permutation)
    } else {
        Err(Error::bounds(format!(
            "{data:?} has less than two dimensions",
        )))
    }
}

/// Inverse two-dimensional Fast Fourier Transform
pub fn ifft2<T, A>(
    data: Array<Complex<T>, A>,
) -> Result<Array<Complex<T>, impl Access<Complex<T>>>, Error>
where
    A: Access<Complex<T>>,
    T: rustfft::FftNum,
    Complex<T>: crate::Complex,
{
    if data.ndim() >= 2 {
        let mut permutation = (0..data.ndim()).into_iter().collect::<Axes>();
        permutation.swap(data.ndim() - 1, data.ndim() - 2);

        data.transpose(permutation.clone())?
            .ifft()?
            .transpose(permutation)?
            .ifft()
    } else {
        Err(Error::bounds(format!(
            "{data:?} has less than two dimensions",
        )))
    }
}

/// Shift the primary frequency component to the center of the given axis, or invert a shift.
pub fn shift<'a, T, X>(
    data: ArrayAccess<T>,
    axis: X,
) -> Result<Array<T, impl Access<T> + 'a>, Error>
where
    T: Number,
    X: Into<Option<usize>>,
{
    let axis = axis.into().unwrap_or_else(|| data.ndim() - 1);

    if axis < data.ndim() {
        let dim = data.shape()[axis];
        let pivot = dim / 2 + 1;

        let range = slice_range(data.shape(), axis, 0..pivot);
        let left = data.clone().slice(range)?;

        let range = slice_range(data.shape(), axis, pivot..dim);
        let right = data.clone().slice(range)?;

        Array::transpose_concat(vec![left, right], axis)
    } else {
        Err(Error::bounds(format!("{data:?} has no axis {axis}")))
    }
}

#[inline]
fn slice_range(shape: &[usize], axis: usize, range: std::ops::Range<usize>) -> Range {
    shape[..axis]
        .into_iter()
        .copied()
        .map(|dim| 0..dim)
        .map(AxisRange::from)
        .chain(iter::once(range.into()))
        .collect()
}
