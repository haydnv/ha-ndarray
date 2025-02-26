use num_complex::Complex;

use crate::{Access, Array, Axes, Error, NDArray, NDArrayFourier, NDArrayTransform};

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
#[cfg(feature = "complex")]
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
        Err(Error::Bounds(format!(
            "array of shape {:?} has less than two dimensions",
            data.shape()
        )))
    }
}

/// Inverse two-dimensional Fast Fourier Transform
#[cfg(feature = "complex")]
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
        Err(Error::Bounds(format!(
            "array of shape {:?} has less than two dimensions",
            data.shape()
        )))
    }
}
