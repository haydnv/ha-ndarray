use ocl::{Buffer, Error, Kernel, Program, Queue};

use crate::CDatatype;

pub fn elementwise_cmp<T: CDatatype>(
    cmp: &'static str,
    queue: Queue,
    left: &Buffer<T>,
    right: &Buffer<T>,
    output: Buffer<u8>,
) -> Result<Buffer<u8>, Error> {
    assert_eq!(left.len(), right.len());
    assert_eq!(left.len(), output.len());

    let src = format!(
        r#"
        __kernel void elementwise_cmp(
            __global {dtype}* left,
            __global {dtype}* right,
            __global char* output)
        {{
            uint const idx = get_global_id(0);
            if (left[idx] {cmp} right[idx]) {{
                output[idx] = 1;
            }} else {{
                output[idx] = 0;
            }}
        }}
    "#,
        dtype = T::TYPE_STR,
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let kernel = Kernel::builder()
        .name("elementwise_cmp")
        .program(&program)
        .queue(queue)
        .global_work_size(output.len())
        .arg(left)
        .arg(right)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}

pub fn elementwise_inplace<T: CDatatype>(
    op: &'static str,
    queue: Queue,
    left: Buffer<T>,
    right: &Buffer<T>,
) -> Result<Buffer<T>, Error> {
    assert_eq!(left.len(), right.len());

    let src = format!(
        r#"
        __kernel void elementwise_inplace(
            __global {dtype}* left,
            __global {dtype}* right)
        {{
            uint const idx = get_global_id(0);
            left[idx] {op} right[idx];
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

    unsafe { kernel.enq()? }

    Ok(left)
}

pub fn reduce_all<T: CDatatype>(queue: Queue, input: Buffer<T>) -> Result<bool, Error> {
    let src = format!(
        r#"
        __kernel void reduce_all(
                __global uint* flag,
                __global {dtype}* input)
        {{
            if (input[get_global_id(0)] == 0) {{
                flag[0] = 0;
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let mut result = vec![1u8];
    let flag: Buffer<u8> = Buffer::builder().queue(queue.clone()).len(1).build()?;
    flag.write(&result).enq()?;

    let kernel = Kernel::builder()
        .name("reduce_all")
        .program(&program)
        .queue(queue)
        .global_work_size(input.len())
        .arg(&flag)
        .arg(&input)
        .build()?;

    unsafe { kernel.enq()? }

    flag.read(&mut result).enq()?;

    Ok(result == [1])
}

pub fn reduce_any<T: CDatatype>(queue: Queue, input: Buffer<T>) -> Result<bool, Error> {
    let src = format!(
        r#"
        __kernel void reduce_any(
                __global uint* flag,
                __global {dtype}* input)
        {{
            if (input[get_global_id(0)] != 0) {{
                flag[0] = 1;
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let mut result = vec![0u8];
    let flag: Buffer<u8> = Buffer::builder().queue(queue.clone()).len(1).build()?;
    flag.write(&result).enq()?;

    let kernel = Kernel::builder()
        .name("reduce_any")
        .program(&program)
        .queue(queue)
        .global_work_size(input.len())
        .arg(&flag)
        .arg(&input)
        .build()?;

    unsafe { kernel.enq()? }

    flag.read(&mut result).enq()?;

    Ok(result == [1])
}
