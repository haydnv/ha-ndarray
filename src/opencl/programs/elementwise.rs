use memoize::memoize;
use ocl::Program;

use crate::Error;

use super::build;

#[memoize]
pub fn dual_boolean(c_type: &'static str, op: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline uchar eq(const {c_type} left, const {c_type} right) {{
            return left == right;
        }}

        inline uchar ge(const {c_type} left, const {c_type} right) {{
            return left >= right;
        }}

        inline uchar gt(const {c_type} left, const {c_type} right) {{
            return left > right;
        }}

        inline uchar le(const {c_type} left, const {c_type} right) {{
            return left <= right;
        }}

        inline uchar lt(const {c_type} left, const {c_type} right) {{
            return left < right;
        }}

        inline uchar ne(const {c_type} left, const {c_type} right) {{
            return left != right;
        }}

        inline uchar and(const {c_type} left, const {c_type} right) {{
            return (left != 0) && (right != 0);
        }}

        inline uchar or(const {c_type} left, const {c_type} right) {{
            return (left != 0) || (right != 0);
        }}

        inline uchar xor(const {c_type} left, const {c_type} right) {{
            return (left != 0) ^ (right != 0);
        }}

        __kernel void dual(
            __global const {c_type}* restrict left,
            __global const {c_type}* restrict right,
            __global uchar* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(left[offset], right[offset]);
        }}

        __kernel void dual_scalar(
            __global const {c_type}* restrict left,
            const {c_type} right,
            __global uchar* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(left[offset], right);
        }}
        "#
    );

    build(&src)
}

#[memoize]
pub fn dual(c_type: &'static str, op: &'static str) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline {c_type} add(const {c_type} left, const {c_type} right) {{
            return left + right;
        }}

        inline {c_type} sub(const {c_type} left, const {c_type} right) {{
            return left - right;
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
    );

    build(&src)
}

pub fn unary(
    f_type: &'static str,
    i_type: &'static str,
    o_type: &'static str,
    op: &'static str,
) -> Result<Program, Error> {
    let src = format!(
        r#"
        inline {f_type} _log(const {f_type} input) {{
            return log(input);
        }}

        __kernel void unary(__global const {i_type}* input, __global {o_type}* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(input[offset]);
        }}
        "#,
    );

    build(&src)
}
