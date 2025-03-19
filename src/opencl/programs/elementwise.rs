use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::{build, Builder, ElementDual, ElementUnary};

#[memoize]
pub fn cast(op: ElementUnary) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void cast(
            __global const {i_type}* restrict input,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(input[offset]);
        }}
        "#,
    );

    build(&src)
}

#[memoize]
pub fn dual(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual(
            __global const {i_type}* restrict left,
            __global const {i_type}* restrict right,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(left[offset], right[offset]);
        }}
        "#,
    );

    build(&src)
}

#[memoize]
pub fn dual_scalar(op: ElementDual) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual_scalar(
            __global const {i_type}* restrict left,
            const {i_type} right,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(left[offset], right);
        }}
        "#,
    );

    build(&src)
}

pub fn unary(op: ElementUnary) -> Result<Program, Error> {
    let i_type = op.i_type;
    let o_type = op.o_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void unary(__global const {i_type}* input, __global {o_type}* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(input[offset]);
        }}
        "#,
    );

    build(&src)
}
