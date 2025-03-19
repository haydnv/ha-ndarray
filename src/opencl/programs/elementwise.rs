use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::{build, Builder, ElementDual, ElementDualBoolean, ElementUnary};

#[memoize]
pub fn cast(i_type: &'static str, o_type: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void cast(
            __global const {i_type}* restrict input,
            __global {o_type}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = ({o_type}) input[offset];
        }}
        "#,
    );

    build(&src)
}

#[memoize]
pub fn dual_boolean(op: ElementDualBoolean) -> Result<Program, Error> {
    let c_type = op.c_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual(
            __global const {c_type}* restrict left,
            __global const {c_type}* restrict right,
            __global uchar* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(left[offset], right[offset]);
        }}
        "#
    );

    build(&src)
}

#[memoize]
pub fn dual_scalar_boolean(op: ElementDualBoolean) -> Result<Program, Error> {
    let c_type = op.c_type;
    let name = op.name;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual_scalar(
            __global const {c_type}* restrict left,
            const {c_type} right,
            __global uchar* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {name}(left[offset], right);
        }}
        "#
    );

    build(&src)
}

#[memoize]
pub fn dual(op: ElementDual) -> Result<Program, Error> {
    let name = op.name;
    let c_type = op.c_type;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual(
            __global const {c_type}* restrict left,
            __global const {c_type}* restrict right,
            __global {c_type}* restrict output)
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
    let name = op.name;
    let c_type = op.c_type;
    let op = op.build();

    let src = format!(
        r#"
        {op}

        __kernel void dual_scalar(
            __global const {c_type}* restrict left,
            const {c_type} right,
            __global {c_type}* restrict output)
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
