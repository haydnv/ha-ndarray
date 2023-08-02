use ocl::{Error, Program};

use crate::{CDatatype, Context};

pub fn gather_cond<T: CDatatype>(context: &Context) -> Result<Program, Error> {
    let src = format!(
        r#"
        __kernel void gather_cond(
            __global const uchar* restrict cond,
            __global const {dtype}* restrict then,
            __global const {dtype}* restrict or_else,
            __global {dtype}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            if (cond[offset] != 0) {{
                output[offset] = then[offset];
            }} else {{
                output[offset] = or_else[offset];
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    Program::builder().source(src).build(context.cl_context())
}
