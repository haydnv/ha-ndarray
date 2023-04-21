use ocl::{Buffer, Error, Event, Kernel, Program, Queue};

use crate::CDatatype;

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

pub fn elementwise_scalar<T: CDatatype>(
    op: &'static str,
    queue: Queue,
    left: Buffer<T>,
    right: T,
) -> Result<Buffer<T>, Error> {
    let src = format!(
        r#"
        __kernel void elementwise_scalar(__global {dtype}* left, const {dtype} right) {{
            const ulong offset = get_global_id(0);
            left[offset] {op}= right;
        }}
        "#,
        dtype = T::TYPE_STR,
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
