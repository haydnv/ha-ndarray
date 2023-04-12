use ocl::{Buffer, Error, Event, Kernel, Program, Queue};

use crate::CDatatype;

// TODO: move to a custom Platform struct
const MIN_SIZE: usize = 1024;

// TODO: is there a good way to determine this at runtime?
const WG_SIZE: usize = 64;

pub fn elementwise_cmp<T: CDatatype>(
    cmp: &'static str,
    queue: Queue,
    left: &Buffer<T>,
    right: &Buffer<T>,
    output: Buffer<u8>,
    ewait: &Event,
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

    unsafe { kernel.cmd().ewait(ewait).enq()? }

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

pub fn reduce_sum<T: CDatatype>(queue: Queue, mut buffer: Buffer<T>) -> Result<T, Error> {
    if buffer.len() < MIN_SIZE {
        let mut result = vec![T::zero(); buffer.len()];
        buffer.read(&mut result).enq()?;
        return Ok(result.into_iter().sum());
    }

    let src = format!(
        r#"
        __kernel void reduce_sum(
                __global {dtype}* input,
                __global {dtype}* output,
                __local {dtype}* partial_sums)
        {{
            uint global_size = get_global_size(0);
            uint group_size = get_local_size(0);

            uint idx = get_global_id(0);
            uint group_idx = idx / group_size;
            uint local_idx = idx % group_size;

            // copy from global to local memory
            partial_sums[local_idx] = input[idx];

            // sum over local memory in parallel
            for (uint stride = group_size >> 1; stride > 0; stride = stride >> 1) {{
                barrier(CLK_LOCAL_MEM_FENCE);
                partial_sums[local_idx] += partial_sums[local_idx + stride];
            }}

            if (local_idx == 0) {{
                output[group_idx] = partial_sums[local_idx];
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let mut tail_sum = T::zero();
    while buffer.len() >= MIN_SIZE {
        let input_size = WG_SIZE * (buffer.len() / WG_SIZE);

        debug_assert!(input_size > 0);
        debug_assert!(input_size / WG_SIZE > 0);
        debug_assert_eq!(input_size % WG_SIZE, 0);

        if input_size < buffer.len() {
            let mut tail = vec![T::zero(); buffer.len() - input_size];

            buffer
                .create_sub_buffer(None, input_size, tail.len())?
                .read(&mut tail)
                .enq()?;

            tail_sum += tail.into_iter().sum();
        }

        let input = buffer;

        let output = Buffer::builder()
            .queue(queue.clone())
            .len(input_size / WG_SIZE)
            .fill_val(T::zero())
            .build()?;

        let kernel = Kernel::builder()
            .name("reduce_sum")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(WG_SIZE)
            .global_work_size(input_size)
            .arg(&input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        buffer = output;
    }

    let mut result = vec![T::zero(); buffer.len()];
    buffer.read(&mut result).enq()?;

    let sum: T = result.into_iter().sum();
    Ok(tail_sum + sum)
}

#[inline]
fn div_ceil(num: usize, denom: usize) -> usize {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}
