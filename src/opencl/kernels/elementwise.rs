use ocl::{Context, Error, Program};

use crate::CType;

pub fn dual<LT, RT>(op: &'static str, context: &Context) -> Result<Program, Error>
where
    LT: CType,
    RT: CType,
{
    let src = format!(
        r#"
        inline {ltype} add(const {ltype} left, const {rtype} right) {{
            return left + right;
        }}

        __kernel void dual(
            __global const {ltype}* restrict left,
            __global const {rtype}* restrict right,
            __global {ltype}* restrict output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = {op}(left[offset], right[offset]);
        }}
        "#,
        ltype = LT::TYPE,
        rtype = RT::TYPE,
    );

    Program::builder().source(src).build(context)
}
