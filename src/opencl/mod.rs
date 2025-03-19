//! The OpenCL platform

use lazy_static::lazy_static;
use ocl::OclPrm;

use crate::access::{AccessBuf, AccessOp};
use crate::host::VEC_MIN_SIZE;
use crate::Number;

use programs::{ElementDual, ElementUnary};

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

fn real_trig<T: Number>(name: &'static str) -> ElementUnary {
    debug_assert!(name.starts_with('_'));
    ElementUnary::new::<T, T::Float, _>(name, format!("return {}(n);", &name[1..]))
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

    // basic arithmetic (unary)
    fn cl_abs() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_abs", "return abs(n);")
    }

    fn cl_exp() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("_exp", "return exp(n);")
    }

    fn cl_ln() -> ElementUnary {
        ElementUnary::new::<Self, Self, _>("ln", "return log(n);")
    }

    fn cl_round() -> Option<ElementUnary> {
        Some(ElementUnary::new::<Self, Self, _>(
            "_round",
            "return round(n));",
        ))
    }

    // basic arithmetic (dual)
    fn cl_add() -> ElementDual {
        ElementDual::new::<Self, Self, _>("add", "return lhs + rhs;")
    }

    fn cl_div() -> ElementDual {
        ElementDual::new::<Self, Self, _>(
            "div",
            "if (rhs == 0) { return 0; } else { return lhs / rhs; }",
        )
    }

    fn cl_log() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_log", "return log(lhs) / log(rhs);")
    }

    fn cl_mul() -> ElementDual {
        ElementDual::new::<Self, Self, _>("mul", "return lhs * rhs;")
    }

    fn cl_sub() -> ElementDual {
        ElementDual::new::<Self, Self, _>("sub", "return lhs - rhs;")
    }

    fn cl_pow() -> ElementDual {
        ElementDual::new::<Self, Self, _>("_pow", "return pow(lhs, rhs);")
    }

    fn cl_rem() -> Option<ElementDual> {
        ElementDual::new::<Self, Self, _>("rem", "return mod(lhs, rhs);").into()
    }

    // boolean logic (unary)
    fn cl_not() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("not", "return if (n == 0) { 1 } else { 0 };")
    }

    // boolean logic (dual)
    fn cl_and() -> ElementDual {
        ElementDual::new::<Self, u8, _>("and", real_bool_cmp("&&"))
    }

    fn cl_or() -> ElementDual {
        ElementDual::new::<Self, u8, _>("or", real_bool_cmp("||"))
    }

    fn cl_xor() -> ElementDual {
        ElementDual::new::<Self, u8, _>("xor", real_bool_cmp("^"))
    }

    // comparison
    fn cl_eq() -> ElementDual {
        ElementDual::new::<Self, u8, _>("eq", real_cmp("=="))
    }

    fn cl_ne() -> ElementDual {
        ElementDual::new::<Self, u8, _>("ne", real_cmp("!="))
    }

    // floating-point identities
    fn cl_inf() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isinf", "return false;")
    }

    fn cl_nan() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isnan", "return false;")
    }
}

pub trait CLElementOrd: CLElement {
    fn cl_round() -> Option<ElementUnary> {
        Some(ElementUnary::new::<Self, Self, _>(
            "_round",
            "return round(n));",
        ))
    }

    // comparison
    fn cl_ge() -> ElementDual {
        ElementDual::new::<Self, u8, _>("ge", real_cmp(">="))
    }

    fn cl_gt() -> ElementDual {
        ElementDual::new::<Self, u8, _>("gt", real_cmp(">"))
    }

    fn cl_le() -> ElementDual {
        ElementDual::new::<Self, u8, _>("le", real_cmp("<="))
    }

    fn cl_lt() -> ElementDual {
        ElementDual::new::<Self, u8, _>("lt", real_cmp("<"))
    }

    fn cl_max() -> ElementDual {
        ElementDual::new::<Self, u8, _>("_max", "return max(lhs, rhs);")
    }

    fn cl_min() -> ElementDual {
        ElementDual::new::<Self, u8, _>("_min", "return min(lhs, rhs);")
    }
}

pub trait CLElementTrig {
    // trigonometry
    fn cl_sin() -> ElementUnary;

    fn cl_asin() -> ElementUnary;

    fn cl_sinh() -> ElementUnary;

    fn cl_cos() -> ElementUnary;

    fn cl_acos() -> ElementUnary;

    fn cl_cosh() -> ElementUnary;

    fn cl_tan() -> ElementUnary;

    fn cl_atan() -> ElementUnary;

    fn cl_tanh() -> ElementUnary;
}

macro_rules! cl_trig_real {
    ($t:ty) => {
        impl CLElementTrig for $t {
            fn cl_sin() -> ElementUnary {
                real_trig::<Self>("_sin")
            }

            fn cl_asin() -> ElementUnary {
                real_trig::<Self>("_asin")
            }

            fn cl_sinh() -> ElementUnary {
                real_trig::<Self>("_sinh")
            }

            fn cl_cos() -> ElementUnary {
                real_trig::<Self>("_cos")
            }

            fn cl_acos() -> ElementUnary {
                real_trig::<Self>("_acos")
            }

            fn cl_cosh() -> ElementUnary {
                real_trig::<Self>("_cosh")
            }

            fn cl_tan() -> ElementUnary {
                real_trig::<Self>("_tan")
            }

            fn cl_atan() -> ElementUnary {
                real_trig::<Self>("_atan")
            }

            fn cl_tanh() -> ElementUnary {
                real_trig::<Self>("_tanh")
            }
        }
    };
}

impl CLElement for f32 {
    const TYPE: &'static str = "float";

    fn cl_rem() -> Option<ElementDual> {
        Some(ElementDual::new::<Self, Self, _>(
            "rem",
            "return fmod(lhs, rhs);",
        ))
    }

    fn cl_inf() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isinf", "return isinf(n);")
    }

    fn cl_nan() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isnan", "return isnan(n);")
    }
}

impl CLElementOrd for f32 {}

cl_trig_real!(f32);

impl CLElement for f64 {
    const TYPE: &'static str = "double";

    fn cl_rem() -> Option<ElementDual> {
        Some(ElementDual::new::<Self, Self, _>(
            "rem",
            "return fmod(lhs, rhs);",
        ))
    }

    fn cl_inf() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isinf", "return isinf(n);")
    }

    fn cl_nan() -> ElementUnary {
        ElementUnary::new::<Self, u8, _>("_isnan", "return isnan(n);")
    }
}

impl CLElementOrd for f64 {}

cl_trig_real!(f64);

impl CLElement for i8 {
    const TYPE: &'static str = "char";
}

impl CLElementOrd for i8 {}

cl_trig_real!(i8);

impl CLElement for i16 {
    const TYPE: &'static str = "short";
}

impl CLElementOrd for i16 {}

cl_trig_real!(i16);

impl CLElement for i32 {
    const TYPE: &'static str = "int";
}

impl CLElementOrd for i32 {}

cl_trig_real!(i32);

impl CLElement for i64 {
    const TYPE: &'static str = "long";
}

impl CLElementOrd for i64 {}

cl_trig_real!(i64);

impl CLElement for u8 {
    const TYPE: &'static str = "uchar";
}

impl CLElementOrd for u8 {}

cl_trig_real!(u8);

impl CLElement for u16 {
    const TYPE: &'static str = "ushort";
}

impl CLElementOrd for u16 {}

cl_trig_real!(u16);

impl CLElement for u32 {
    const TYPE: &'static str = "uint";
}

impl CLElementOrd for u32 {}

cl_trig_real!(u32);

impl CLElement for u64 {
    const TYPE: &'static str = "ulong";
}

impl CLElementOrd for u64 {}

cl_trig_real!(u64);

#[cfg(feature = "complex")]
macro_rules! cl_complex {
    ($t:ty, $ct:expr) => {
        impl CLElement for num_complex::Complex<$t> {
            const TYPE: &'static str = $ct;

            // basic arithmetic (dual)
            fn cl_div() -> ElementDual {
                ElementDual::new::<Self, Self, _>("div", complex_div::<Self>())
            }

            fn cl_mul() -> ElementDual {
                ElementDual::new::<Self, Self, _>("mul", complex_mul::<Self>())
            }

            fn cl_rem() -> Option<ElementDual> {
                None
            }

            // boolean logic
            fn cl_and() -> ElementDual {
                ElementDual::new::<Self, u8, _>("and", complex_bool_cmp("&&"))
            }

            fn cl_or() -> ElementDual {
                ElementDual::new::<Self, u8, _>("and", complex_bool_cmp("||"))
            }

            fn cl_xor() -> ElementDual {
                ElementDual::new::<Self, u8, _>("and", complex_bool_cmp("^"))
            }

            // comparison
            fn cl_eq() -> ElementDual {
                ElementDual::new::<Self, u8, _>("eq", complex_cmp("==", "&&"))
            }

            fn cl_ne() -> ElementDual {
                ElementDual::new::<Self, u8, _>("ne", complex_cmp("!=", "||"))
            }
        }
    };
}

#[cfg(feature = "complex")]
cl_complex!(f32, "float2");
#[cfg(feature = "complex")]
cl_complex!(f64, "double2");

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
