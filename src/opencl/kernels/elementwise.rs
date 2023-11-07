use ocl::{Context, Error, Program};

use crate::CType;

pub fn compare<T: CType>(op: &'static str, context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline uchar eq(const {c_type} left, const {c_type} right) {{
            return left == right;
        }}

        __kernel void compare(
            __global const {c_type}* restrict left,
            __global const {c_type}* restrict right,
            __global uchar* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(left[offset], right[offset]);
        }}
        "#,
        c_type = T::TYPE,
    );

    Program::builder().source(src).build(context)
}

pub fn dual<T: CType>(op: &'static str, context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline {c_type} add(const {c_type} left, const {c_type} right) {{
            return left + right;
        }}

        __kernel void dual(
            __global const {c_type}* restrict left,
            __global const {c_type}* restrict right,
            __global {c_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(left[offset], right[offset]);
        }}
        "#,
        c_type = T::TYPE,
    );

    Program::builder().source(src).build(context)
}
