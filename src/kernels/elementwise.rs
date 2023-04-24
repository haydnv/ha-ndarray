use ocl::{Buffer, Error, Event, Kernel, Program, Queue};

use crate::CDatatype;

pub fn cast<I: CDatatype, O: CDatatype>(
    queue: Queue,
    input: &Buffer<I>,
) -> Result<Buffer<O>, Error> {
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
        itype = I::TYPE_STR,
        otype = O::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(input.len())
        .build()?;

    let kernel = Kernel::builder()
        .name("cast")
        .program(&program)
        .queue(queue)
        .global_work_size(input.len())
        .arg(input)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? };

    Ok(output)
}

pub fn elementwise_boolean<T: CDatatype>(
    cmp: &'static str,
    queue: Queue,
    left: &Buffer<T>,
    right: &Buffer<T>,
    ewait: &Event,
) -> Result<Buffer<u8>, Error> {
    assert_eq!(left.len(), right.len());

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

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(left.len())
        .build()?;

    let kernel = Kernel::builder()
        .name("elementwise_boolean")
        .program(&program)
        .queue(queue)
        .global_work_size(output.len())
        .arg(left)
        .arg(right)
        .arg(&output)
        .build()?;

    unsafe { kernel.cmd().ewait(ewait).enq()? }

    Ok(output)
}

pub fn elementwise_cmp<T: CDatatype>(
    cmp: &'static str,
    queue: Queue,
    left: &Buffer<T>,
    right: &Buffer<T>,
    ewait: &Event,
) -> Result<Buffer<u8>, Error> {
    assert_eq!(left.len(), right.len());

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

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(left.len())
        .build()?;

    let kernel = Kernel::builder()
        .name("elementwise_cmp")
        .program(&program)
        .queue(queue)
        .global_work_size(output.len())
        .arg(left)
        .arg(right)
        .arg(&output)
        .build()?;

    unsafe { kernel.cmd().ewait(ewait).enq()? }

    Ok(output)
}

pub fn elementwise_inplace<T: CDatatype>(
    op: &'static str,
    queue: Queue,
    left: Buffer<T>,
    right: &Buffer<T>,
    ewait: &Event,
) -> Result<Buffer<T>, Error> {
    assert_eq!(left.len(), right.len());

    let src = format!(
        r#"
        __kernel void elementwise_inplace(
            __global {dtype}* restrict left,
            __global const {dtype}* restrict right)
        {{
            const ulong offset = get_global_id(0);
            left[offset] {op}= right[offset];
        }}
        "#,
        dtype = T::TYPE_STR,
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let kernel = Kernel::builder()
        .name("elementwise_inplace")
        .program(&program)
        .queue(queue)
        .global_work_size(left.len())
        .arg(&left)
        .arg(right)
        .build()?;

    unsafe { kernel.cmd().ewait(ewait).enq()? }

    Ok(left)
}

// TODO: optimize to use integer modulo and +=, -=, etc. when possible
pub fn elementwise_scalar<IT, OT>(
    op: &'static str,
    queue: Queue,
    left: Buffer<OT>,
    right: IT,
) -> Result<Buffer<OT>, Error>
where
    IT: CDatatype,
    OT: CDatatype,
{
    let src = format!(
        r#"
        inline {otype} add({otype} left, const {itype} right) {{
            return left + right;
        }}

        inline {otype} div({otype} left, const {itype} right) {{
            return left / right;
        }}

        inline {otype} mul({otype} left, const {itype} right) {{
            return left * right;
        }}

        inline {otype} pow_({otype} left, const double right) {{
            return pow((double) left, right);
        }}

        inline {otype} sub({otype} left, const {itype} right) {{
            return left - right;
        }}

        __kernel void elementwise_scalar(__global {otype}* left, const {itype} right) {{
            const ulong offset = get_global_id(0);
            left[offset] = {op}(left[offset], right);
        }}
        "#,
        itype = IT::TYPE_STR,
        otype = OT::TYPE_STR,
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let kernel = Kernel::builder()
        .name("elementwise_scalar")
        .program(&program)
        .queue(queue)
        .global_work_size(left.len())
        .arg(&left)
        .arg(right)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(left)
}

pub fn scalar_cmp<T: CDatatype>(
    cmp: &'static str,
    queue: Queue,
    input: &Buffer<T>,
    scalar: T,
) -> Result<Buffer<u8>, Error> {
    let src = format!(
        r#"
        __kernel void scalar_cmp(
            __global const {dtype}* input,
            __private const {dtype} right,
            __global uchar* output)
        {{
            const ulong offset = get_global_id(0);
            output[offset] = input[offset] {cmp} right;
        }}
        "#,
        dtype = T::TYPE_STR,
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(input.len())
        .build()?;

    let kernel = Kernel::builder()
        .name("scalar_cmp")
        .program(&program)
        .queue(queue)
        .global_work_size(input.len())
        .arg(input)
        .arg(scalar)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}

pub fn unary<T: CDatatype>(
    op: &'static str,
    queue: Queue,
    buffer: Buffer<T>,
) -> Result<Buffer<T>, Error> {
    let src = format!(
        r#"
        __kernel void unary(__global {dtype}* buffer) {{
            const ulong offset = get_global_id(0);
            buffer[offset] = {op}(buffer[offset]);
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let kernel = Kernel::builder()
        .name("unary")
        .program(&program)
        .queue(queue)
        .global_work_size(buffer.len())
        .arg(&buffer)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(buffer)
}
