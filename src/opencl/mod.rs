//! The OpenCL platform

use lazy_static::lazy_static;
use ocl::OclPrm;

use crate::access::{AccessBuf, AccessOp};
use crate::host::VEC_MIN_SIZE;

use programs::{ElementDual, ElementDualBoolean};

pub use buffer::*;
pub use platform::{OpenCL, ACC_MIN_SIZE, GPU_MIN_SIZE};

mod buffer;
pub mod ops;
mod platform;
mod programs;

const TILE_SIZE: usize = 8;

const WG_SIZE: usize = 64;

fn real_bool(name: &'static str) -> String {
    format!("{name} != 0")
}

fn real_bool_cmp(op: &'static str) -> String {
    format!(
        "return ({lhs}) {op} ({rhs});",
        lhs = real_bool("lhs"),
        rhs = real_bool("rhs")
    )
}

fn real_cmp(op: &'static str) -> String {
    format!("return lhs {op} rhs;")
}

#[cfg(feature = "complex")]
fn complex_bool(name: &'static str) -> String {
    format!("{name}.x != 0 || {name}.y != 0")
}

#[cfg(feature = "complex")]
fn complex_bool_cmp(op: &'static str) -> String {
    format!(
        "return ({lhs}) {op} ({rhs});",
        lhs = complex_bool("lhs"),
        rhs = complex_bool("rhs")
    )
}

#[cfg(feature = "complex")]
fn complex_cmp(cmp: &'static str, cond: &'static str) -> String {
    format!("return (lhs.x {cmp} rhs.x) {cond} (lhs.y {cmp} rhs.y);")
}

#[cfg(feature = "complex")]
fn complex_div<T: CLElement>() -> String {
    format!(
        "
        if (rhs.x == 0.0f && rhs.y == 0.0f) {{
            return ({c_type})(0.0f, 0.0f);
        }} else {{
            float denom = (rhs.x * rhs.x) + (rhs.y * rhs.y);
            float re = ((lhs.x * rhs.x) + (lhs.y * rhs.y)) / denom;
            float im = ((lhs.y * rhs.x) - (lhs.x * rhs.y)) / denom;
            return ({c_type})(re, im);
        }}",
        c_type = T::TYPE,
    )
}

#[cfg(feature = "complex")]
fn complex_mul<T: CLElement>() -> String {
    format!(
        "
        float re = ((lhs.x * rhs.x) - (lhs.y * rhs.y));
        float im = ((lhs.x * rhs.y) + (lhs.y * rhs.x));
        return ({c_type})(re, im);
        ",
        c_type = T::TYPE,
    )
}

pub trait CLElement: OclPrm {
    const TYPE: &'static str;

    // basic arithmetic (dual)
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, _>("add", "return lhs + rhs;")
    }

    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, _>(
            "div",
            "if (rhs == 0) { return 0; } else { return lhs / rhs; }",
        )
    }

    fn cl_log() -> ElementDual {
        ElementDual::new::<Self, _>("_log", "return log(lhs) / log(rhs);")
    }

    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, _>("mul", "return lhs * rhs;")
    }

    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, _>("sub", "return lhs - rhs;")
    }

    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, _>("_pow", "return pow(lhs, rhs);")
    }

    fn cl_rem() -> Option<ElementDual> {
        ElementDual::new::<Self, _>("rem", "return mod(lhs, rhs);").into()
    }

    // boolean logic
    fn cl_and() -> ElementDualBoolean {
        ElementDualBoolean::new::<Self, _>("and", real_bool_cmp("&&"))
    }

    fn cl_or() -> ElementDualBoolean {
        ElementDualBoolean::new::<Self, _>("or", real_bool_cmp("||"))
    }

    fn cl_xor() -> ElementDualBoolean {
        ElementDualBoolean::new::<Self, _>("xor", real_bool_cmp("^"))
    }

    // comparison
    fn cl_eq() -> ElementDualBoolean {
        ElementDualBoolean::new::<Self, _>("eq", real_cmp("=="))
    }

    fn cl_ne() -> ElementDualBoolean {
        ElementDualBoolean::new::<Self, _>("ne", real_cmp("!="))
    }

    fn cl_ge() -> Option<ElementDualBoolean> {
        Some(ElementDualBoolean::new::<Self, _>("ge", real_cmp(">=")))
    }

    fn cl_gt() -> Option<ElementDualBoolean> {
        Some(ElementDualBoolean::new::<Self, _>("gt", real_cmp(">")))
    }

    fn cl_le() -> Option<ElementDualBoolean> {
        Some(ElementDualBoolean::new::<Self, _>("le", real_cmp("<=")))
    }

    fn cl_lt() -> Option<ElementDualBoolean> {
        Some(ElementDualBoolean::new::<Self, _>("lt", real_cmp("<")))
    }
}

impl CLElement for f32 {
    const TYPE: &'static str = "float";

    fn cl_rem() -> Option<ElementDual> {
        Some(ElementDual::new::<Self, _>("rem", "return fmod(lhs, rhs);"))
    }
}

impl CLElement for f64 {
    const TYPE: &'static str = "double";

    fn cl_rem() -> Option<ElementDual> {
        Some(ElementDual::new::<Self, _>("rem", "return fmod(lhs, rhs);"))
    }
}

impl CLElement for i8 {
    const TYPE: &'static str = "char";
}

impl CLElement for i16 {
    const TYPE: &'static str = "short";
}

impl CLElement for i32 {
    const TYPE: &'static str = "int";
}

impl CLElement for i64 {
    const TYPE: &'static str = "long";
}

impl CLElement for u8 {
    const TYPE: &'static str = "uchar";
}

impl CLElement for u16 {
    const TYPE: &'static str = "ushort";
}

impl CLElement for u32 {
    const TYPE: &'static str = "uint";
}

impl CLElement for u64 {
    const TYPE: &'static str = "ulong";
}

#[cfg(feature = "complex")]
macro_rules! complex_cl {
    ($t:ty, $ct:expr) => {
        impl CLElement for $t {
            const TYPE: &'static str = $ct;

            // basic arithmetic (dual)
            fn cl_div() -> ElementDual {
                ElementDual::new::<Self, _>("div", complex_div::<Self>())
            }

            fn cl_mul() -> ElementDual {
                ElementDual::new::<Self, _>("mul", complex_mul::<Self>())
            }

            fn cl_rem() -> Option<ElementDual> {
                None
            }

            // boolean logic
            fn cl_and() -> ElementDualBoolean {
                ElementDualBoolean::new::<Self, _>("and", complex_bool_cmp("&&"))
            }

            fn cl_or() -> ElementDualBoolean {
                ElementDualBoolean::new::<Self, _>("and", complex_bool_cmp("||"))
            }

            fn cl_xor() -> ElementDualBoolean {
                ElementDualBoolean::new::<Self, _>("and", complex_bool_cmp("^"))
            }

            // comparison
            fn cl_eq() -> ElementDualBoolean {
                ElementDualBoolean::new::<Self, _>("eq", complex_cmp("==", "&&"))
            }

            fn cl_ne() -> ElementDualBoolean {
                ElementDualBoolean::new::<Self, _>("ne", complex_cmp("!=", "||"))
            }

            fn cl_ge() -> Option<ElementDualBoolean> {
                None
            }

            fn cl_gt() -> Option<ElementDualBoolean> {
                None
            }

            fn cl_le() -> Option<ElementDualBoolean> {
                None
            }

            fn cl_lt() -> Option<ElementDualBoolean> {
                None
            }
        }
    };
}

#[cfg(feature = "complex")]
complex_cl!(num_complex::Complex<f32>, "float2");
#[cfg(feature = "complex")]
complex_cl!(num_complex::Complex<f64>, "double2");

lazy_static! {
    pub static ref CL_PLATFORM: platform::CLPlatform = {
        assert!(VEC_MIN_SIZE < GPU_MIN_SIZE);
        assert!(GPU_MIN_SIZE < ACC_MIN_SIZE);

        platform::CLPlatform::default().expect("OpenCL platform")
    };
}

pub type ArrayBuf<T> = crate::array::Array<T, AccessBuf<ocl::Buffer<T>>, OpenCL>;
pub type ArrayOp<T, O> = crate::array::Array<T, AccessOp<O, OpenCL>, OpenCL>;

#[cfg(test)]
mod tests {
    use crate::{
        shape, slice, AxisRange, Error, MatrixDual, NDArray, NDArrayCompare, NDArrayMath,
        NDArrayRead, NDArrayReduceBoolean, NDArrayTransform, NDArrayWrite, Shape,
    };

    use super::*;

    #[test]
    fn test_add() -> Result<(), Error> {
        let shape = shape![1, 2, 3];

        let left = ArrayBuf::constant(0, shape.clone())?;
        let right = ArrayBuf::constant(0, shape.clone())?;
        let expected = ArrayBuf::constant(0, shape.clone())?;

        let actual = left.add(right)?;
        let eq = actual.eq(expected)?;

        assert!(eq.all()?);

        Ok(())
    }

    #[test]
    fn test_matmul_2x2() -> Result<(), Error> {
        let l = ArrayOp::range(0, 4, shape![2, 2])?;
        let r = ArrayOp::range(0, 4, shape![2, 2])?;

        let actual = l.matmul(r)?;
        assert_eq!(actual.shape(), &[2, 2]);

        let expected = vec![2, 3, 6, 11];
        assert_eq!(actual.buffer()?.to_slice()?.to_vec(), expected);

        Ok(())
    }

    #[test]
    fn test_matmul_12x20() -> Result<(), Error> {
        let buf = OpenCL::copy_into_buffer::<i32>(&(0..12).into_iter().collect::<Vec<_>>())?;
        let l = ArrayBuf::new(buf, shape![3, 4])?;

        let buf = OpenCL::copy_into_buffer::<i32>(&(0..20).into_iter().collect::<Vec<_>>())?;
        let r = ArrayBuf::new(buf, shape![4, 5])?;

        let actual = l.matmul(r)?;
        assert_eq!(actual.shape(), &[3, 5]);

        let expected = vec![
            70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462,
        ];

        assert_eq!(actual.buffer()?.to_slice()?.to_vec(), expected);

        Ok(())
    }

    #[test]
    fn test_matmul_large() -> Result<(), Error> {
        let shapes: Vec<(Shape, Shape, Shape)> = vec![
            (shape![2, 3], shape![3, 4], shape![2, 4]),
            (shape![9, 7], shape![7, 12], shape![9, 12]),
            (shape![16, 8], shape![8, 24], shape![16, 24]),
            (shape![2, 9], shape![9, 1], shape![2, 1]),
            (shape![16, 8], shape![8, 32], shape![16, 32]),
            (shape![2, 15, 26], shape![2, 26, 37], shape![2, 15, 37]),
            (shape![3, 15, 26], shape![3, 26, 37], shape![3, 15, 37]),
            (shape![8, 44, 1], shape![8, 1, 98], shape![8, 44, 98]),
        ];

        let queue = OpenCL::queue(GPU_MIN_SIZE, &[])?;

        for (left_shape, right_shape, output_shape) in shapes {
            let left = ocl::Buffer::builder()
                .queue(queue.clone())
                .len(left_shape.iter().product::<usize>())
                .fill_val(1.)
                .build()?;

            let right = ocl::Buffer::builder()
                .queue(queue.clone())
                .len(right_shape.iter().product::<usize>())
                .fill_val(1.)
                .build()?;

            let left = ArrayBuf::new(left, left_shape)?;
            let right = ArrayBuf::new(right, right_shape)?;

            let expected = *left.shape().last().unwrap();

            let actual = left.matmul(right)?;
            assert_eq!(actual.shape(), output_shape.as_slice());

            let actual = actual.buffer()?.to_slice()?;
            assert!(
                actual.iter().copied().all(|n| n == expected as f32),
                "expected {expected} but found {actual:?}"
            );

            queue.flush()?;
        }

        Ok(())
    }

    #[test]
    fn test_sub() -> Result<(), Error> {
        let shape = shape![1, 2, 3];

        let buffer = OpenCL::copy_into_buffer::<i32>(&[0, 1, 2, 3, 4, 5])?;
        let array = ArrayBuf::new(buffer, shape.clone())?;

        let actual = array.as_ref().sub(array.as_ref())?;

        assert!(!actual.any()?);

        Ok(())
    }

    #[test]
    fn test_slice() -> Result<(), Error> {
        let buf = OpenCL::copy_into_buffer::<u32>(&[0; 6])?;
        let array = ArrayBuf::new(buf, shape![2, 3])?;
        let mut slice = array.slice(slice![AxisRange::In(0, 2, 1), AxisRange::At(1)])?;

        let buf = OpenCL::copy_into_buffer::<u32>(&[0, 0])?;
        let zeros = ArrayBuf::new(buf, shape![2])?;

        let buf = OpenCL::copy_into_buffer::<u32>(&[0, 0])?;
        let ones = ArrayBuf::new(buf, shape![2])?;

        assert!(slice.as_ref().eq(zeros)?.all()?);

        slice.write(&ones)?;

        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_mul_complex() -> Result<(), Error> {
        type C32 = num_complex::Complex<f32>;

        let buf = OpenCL::copy_into_buffer(&[C32::new(0.5, 0.5)])?;
        let lhs = ArrayBuf::new(buf, shape![1])?;

        let buf = OpenCL::copy_into_buffer(&[C32::new(1., -1.)])?;
        let rhs = ArrayBuf::new(buf, shape![1])?;

        let actual = lhs.mul(rhs)?;
        let actual = actual.buffer()?.to_slice()?;
        assert_eq!(actual.into_vec(), vec![C32::new(1., 0.)]);
        Ok(())
    }

    #[cfg(feature = "complex")]
    #[test]
    fn test_div_complex() -> Result<(), Error> {
        use num_complex::Complex64 as C64;

        let buf = OpenCL::copy_into_buffer(&[C64::new(0.5, 0.5)])?;
        let lhs = ArrayBuf::new(buf, shape![1])?;

        let buf = OpenCL::copy_into_buffer(&[C64::new(1., -1.)])?;
        let rhs = ArrayBuf::new(buf, shape![1])?;

        let actual = lhs.div(rhs)?;
        let actual = actual.buffer()?.to_slice()?;
        assert_eq!(actual.into_vec(), vec![C64::new(0., 0.5)]);
        Ok(())
    }
}
