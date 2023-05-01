use ocl::{Error, Program};

use crate::{CDatatype, Context};

pub fn cast<IT, OT>(context: &Context) -> Result<Program, Error>
where
    IT: CDatatype,
    OT: CDatatype,
{
    let src = format!(
        r#"
        __kernel void cast(
            __global const {itype}* restrict input,
            __global {otype}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = ({otype}) input[offset];
        }}
        "#,
        itype = IT::TYPE_STR,
        otype = OT::TYPE_STR
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn elementwise_boolean<T>(cmp: &'static str, context: &Context) -> Result<Program, Error>
where
    T: CDatatype,
{
    let src = format!(
        r#"
        __kernel void elementwise_boolean(
            __global const {dtype}* restrict left,
            __global const {dtype}* restrict right,
            __global uchar* output)
        {{
            const ulong offset = get_global_id(0);
            const bool left_bool = left[offset] != 0;
            const bool right_bool = right[offset] != 0;

            if (left_bool {cmp} right_bool) {{
                output[offset] = 1;
            }} else {{
                output[offset] = 0;
            }}
        }}
        "#,
        dtype = T::TYPE_STR,
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn elementwise_cmp<T>(cmp: &'static str, context: &Context) -> Result<Program, Error>
where
    T: CDatatype,
{
    let src = format!(
        r#"
        __kernel void elementwise_cmp(
            __global const {dtype}* restrict left,
            __global const {dtype}* restrict right,
            __global uchar* output)
        {{
            const ulong offset = get_global_id(0);

            if (left[offset] {cmp} right[offset]) {{
                output[offset] = 1;
            }} else {{
                output[offset] = 0;
            }}
        }}
        "#,
        dtype = T::TYPE_STR,
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn elementwise_inplace<LT, RT>(op: &'static str, context: &Context) -> Result<Program, Error>
where
    LT: CDatatype,
    RT: CDatatype,
{
    let src = format!(
        r#"
        inline void add({ltype}* left, const {rtype} right) {{
            *left += right;
        }}

        inline void log_({ltype}* left, const double right) {{
            *left = log((double) *left) / log(right);
        }}

        inline void div({ltype}* left, const {rtype} right) {{
            *left /= right;
        }}

        inline void mul({ltype}* left, const {rtype} right) {{
            *left *= right;
        }}

        inline void pow_({ltype}* left, const double right) {{
            *left = pow((double) *left, right);
        }}

        inline void sub({ltype}* left, const {rtype} right) {{
            *left -= right;
        }}

        __kernel void elementwise_inplace(
            __global {ltype}* restrict left,
            __global const {rtype}* restrict right)
        {{
            const ulong offset = get_global_id(0);
            {op}(&left[offset], right[offset]);
        }}
        "#,
        ltype = LT::TYPE_STR,
        rtype = RT::TYPE_STR,
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn elementwise_scalar<IT, OT>(op: &'static str, context: &Context) -> Result<Program, Error>
where
    IT: CDatatype,
    OT: CDatatype,
{
    let src = format!(
        r#"
        inline void add({otype}* left, const {itype} right) {{
            *left += right;
        }}

        inline void div({otype}* left, const {itype} right) {{
            *left /= right;
        }}

        inline void mul({otype}* left, const {itype} right) {{
            *left *= right;
        }}

        inline void pow_({otype}* left, const double right) {{
            *left = pow((double) *left, right);
        }}

        inline void sub({otype}* left, const {itype} right) {{
            *left -= right;
        }}

        __kernel void elementwise_scalar(__global {otype}* left, const {itype} right) {{
            const ulong offset = get_global_id(0);
            {op}(left[offset], right);
        }}
        "#,
        itype = IT::TYPE_STR,
        otype = OT::TYPE_STR,
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn scalar_cmp<T: CDatatype>(cmp: &'static str, context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void scalar_cmp(
            __global const {dtype}* input,
            const {dtype} right,
            __global uchar* output)
        {{
            const ulong offset = get_global_id(0);
            if (input[offset] {cmp} right) {{
                output[offset] = 1;
            }} else {{
                output[offset] = 0;
            }}
        }}
        "#,
        dtype = T::TYPE_STR,
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn unary<IT, OT>(op: &'static str, context: &Context) -> Result<Program, Error>
where
    IT: CDatatype,
    OT: CDatatype,
{
    let src = format!(
        r#"
        __kernel void unary(__global const {itype}* input, __global {otype}* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(input[offset]);
        }}
        "#,
        itype = IT::TYPE_STR,
        otype = OT::TYPE_STR,
    );

    Program::builder().source(src).build(context.cl_context())
}
