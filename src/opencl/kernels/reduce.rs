use ocl::{Context, Error, Program};

use crate::CType;

pub fn all<T: CType>(context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void all(
                __global uchar* flag,
                __global const {c_type}* input)
        {{
            if (input[get_global_id(0)] == {zero}) {{
                flag[0] = 0;
            }}
        }}
        "#,
        c_type = T::TYPE,
        zero = T::ZERO,
    );

    Program::builder().source(src).build(context)
}

pub fn any<T: CType>(context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void any(
                __global uchar* flag,
                __global const {c_type}* input)
        {{
            if (input[get_global_id(0)] != {zero}) {{
                flag[0] = 1;
            }}
        }}
        "#,
        c_type = T::TYPE,
        zero = T::ZERO,
    );

    Program::builder().source(src).build(context)
}
