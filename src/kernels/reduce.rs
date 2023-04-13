use ocl::{Buffer, Error, Kernel, Program, Queue};

use crate::{CDatatype, Shape};

// TODO: move to a custom Platform struct
const MIN_SIZE: usize = 1024;

// TODO: is there a good way to determine this at runtime?
const WG_SIZE: usize = 64;

pub fn reduce_all<T: CDatatype>(queue: Queue, input: Buffer<T>) -> Result<bool, Error> {
    let src = format!(
        r#"
        __kernel void reduce_all(
                __global uint* flag,
                __global const {dtype}* input)
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
                __global const {dtype}* input)
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

pub fn reduce<T: CDatatype>(
    init: T,
    reduce: &'static str,
    queue: Queue,
    mut buffer: Buffer<T>,
    collector: impl Fn(std::vec::IntoIter<T>) -> T,
) -> Result<T, Error> {
    if buffer.len() < MIN_SIZE {
        let mut result = vec![init; buffer.len()];
        buffer.read(&mut result).enq()?;
        return Ok((collector)(result.into_iter()));
    }

    let src = format!(
        r#"
        __kernel void reduce(
                const ulong size,
                __global const {dtype}* input,
                __global {dtype}* output,
                __local {dtype}* partials)
        {{
            const uint idx = get_global_id(0);

            if (idx >= size) {{
                return;
            }}

            const uint group_size = get_local_size(0);
            const uint group_idx = idx / group_size;
            const uint local_idx = idx % group_size;

            // copy from global to local memory
            partials[local_idx] = input[idx];

            // reduce over local memory in parallel
            for (uint stride = group_size >> 1; stride > 0; stride = stride >> 1) {{
                barrier(CLK_LOCAL_MEM_FENCE);
                if (idx + stride < size) {{
                    partials[local_idx] {reduce} partials[local_idx + stride];
                }}
            }}

            if (local_idx == 0) {{
                output[group_idx] = partials[local_idx];
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    while buffer.len() >= MIN_SIZE {
        let input = buffer;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(div_ceil(input.len(), WG_SIZE))
            .fill_val(init)
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(WG_SIZE * output.len())
            .arg(input.len())
            .arg(&input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        buffer = output;
    }

    let mut result = vec![init; buffer.len()];
    buffer.read(&mut result).enq()?;
    Ok((collector)(result.into_iter()))
}

pub fn reduce_axis<T: CDatatype>(
    init: T,
    reduce: &'static str,
    queue: Queue,
    input: Buffer<T>,
    shape: Shape,
    axis: usize,
    output: Buffer<T>,
) -> Result<Buffer<T>, Error> {
    assert!(axis < shape.len());

    let src = format!(
        r#"
        __kernel void reduce_axis(
                const ulong input_size,
                const ulong reduce_dim,
                {dtype} init,
                __global const {dtype}* input,
                __global {dtype}* output)
        {{
            const uint idx = get_global_id(0);

            {dtype} reduced = init;
            for (uint i = idx; i < idx + reduce_dim; i++) {{
                reduced {reduce} input[i];
            }}

            output[idx] = reduced;
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let reduce_dim = shape[axis];

    debug_assert_eq!(output.len() * reduce_dim, input.len());

    let kernel = Kernel::builder()
        .name("reduce_axis")
        .program(&program)
        .queue(queue.clone())
        .global_work_size(WG_SIZE * div_ceil(input.len(), WG_SIZE))
        .arg(input.len() as u64)
        .arg(reduce_dim as u64)
        .arg(init)
        .arg(&input)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}

#[inline]
fn div_ceil(num: usize, denom: usize) -> usize {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}
