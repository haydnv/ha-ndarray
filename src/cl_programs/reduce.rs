use ocl::{Buffer, Error, Kernel, Program, Queue};
use rayon::prelude::*;

use crate::{div_ceil, CDatatype};

use super::WG_SIZE;

pub fn reduce_all<T: CDatatype>(queue: Queue, input: &Buffer<T>) -> Result<bool, Error> {
    let src = format!(
        r#"
        __kernel void reduce_all(
                __global uchar* flag,
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
        .queue(queue.clone())
        .global_work_size(input.len())
        .arg(&flag)
        .arg(input)
        .build()?;

    unsafe { kernel.enq()? }

    flag.read(&mut result).enq()?;

    queue.finish()?;

    Ok(result == [1])
}

pub fn reduce_any<T: CDatatype>(queue: Queue, input: &Buffer<T>) -> Result<bool, Error> {
    let src = format!(
        r#"
        __kernel void reduce_any(
                __global uchar* flag,
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
        .queue(queue.clone())
        .global_work_size(input.len())
        .arg(&flag)
        .arg(input)
        .build()?;

    unsafe { kernel.enq()? }

    flag.read(&mut result).enq()?;

    queue.finish()?;

    Ok(result == [1])
}

pub fn reduce<T: CDatatype>(
    init: T,
    reduce: &'static str,
    queue: Queue,
    input: &Buffer<T>,
    collector: impl Fn(T, T) -> T + Send + Sync,
) -> Result<T, Error> {
    const MIN_SIZE: usize = 65_536;

    let min_size = MIN_SIZE * num_cpus::get();

    if input.len() < min_size {
        let mut result = vec![init; input.len()];
        input.read(&mut result).enq()?;
        return Ok(result.into_par_iter().reduce(|| init, collector));
    }

    let src = format!(
        r#"
        inline void add({dtype}* left, const {dtype} right) {{
            *left += right;
        }}

        inline void mul({dtype}* left, const {dtype} right) {{
            *left *= right;
        }}

        __kernel void reduce(
                const ulong size,
                __global const {dtype}* input,
                __global {dtype}* output,
                __local {dtype}* partials)
        {{
            const ulong offset = get_global_id(0);
            const uint group_size = get_local_size(0);
            const ulong a = offset / group_size;
            const uint b = offset % group_size;

            // copy from global to local memory
            partials[b] = input[offset];

            // reduce over local memory in parallel
            for (uint stride = group_size >> 1; stride > 0; stride = stride >> 1) {{
                barrier(CLK_LOCAL_MEM_FENCE);

                if (offset + stride < size) {{
                    uint next = b + stride;
                    if (next < group_size) {{
                        {reduce}(&partials[b], partials[b + stride]);
                    }}
                }}
            }}

            if (b == 0) {{
                output[a] = partials[b];
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let mut buffer = {
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
            .arg(input.len() as u64)
            .arg(input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? };

        output
    };

    while buffer.len() >= min_size {
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
            .arg(input.len() as u64)
            .arg(&input)
            .arg(&output)
            .arg_local::<T>(WG_SIZE)
            .build()?;

        unsafe { kernel.enq()? }

        buffer = output;
    }

    let mut result = vec![init; buffer.len()];
    buffer.read(&mut result).enq()?;

    queue.finish()?;

    Ok(result.into_par_iter().reduce(|| init, collector))
}

pub fn reduce_axis<T: CDatatype>(
    init: T,
    reduce: &'static str,
    queue: Queue,
    input: &Buffer<T>,
    shape: &[usize],
    mut stride: usize,
) -> Result<Buffer<T>, Error> {
    assert!(input.len() > 0);
    assert!(stride > 0);
    assert_eq!(input.len(), shape.iter().product());
    assert_eq!(input.len() % stride, 0);

    let output_size = input.len() / stride;
    debug_assert!(output_size > 0);

    if stride < WG_SIZE {
        return fold_axis(init, reduce, queue.clone(), input, stride, 1);
    }

    let log = (stride as f32).log(WG_SIZE as f32).fract();
    let target_dim = WG_SIZE.pow(log as u32);
    let mut buffer = fold_axis(init, reduce, queue.clone(), input, stride, target_dim)?;

    stride = target_dim;
    debug_assert_eq!(output_size * stride, buffer.len());

    let src = format!(
        r#"
        inline void add({dtype}* left, const {dtype} right) {{
            *left += right;
        }}

        inline void mul({dtype}* left, const {dtype} right) {{
            *left *= right;
        }}

        __kernel void reduce(
                {dtype} init,
                __global const {dtype}* input,
                __global {dtype}* output,
                __local {dtype}* partials)
        {{
            const ulong offset = get_global_id(0);
            const uint reduce_dim = get_local_size(0);

            const ulong a = offset / reduce_dim;
            const uint b = offset % reduce_dim;

            // copy from global to local memory
            partials[b] = input[offset];

            // reduce over local memory in parallel
            for (uint stride = reduce_dim >> 1; stride > 0; stride = stride >> 1) {{
                barrier(CLK_LOCAL_MEM_FENCE);

                uint next = b + stride;
                if (next < reduce_dim) {{
                    {reduce}(&partials[b], partials[next]);
                }}
            }}

            if (b == 0) {{
                output[a] = partials[b];
            }}
        }}
        "#,
        dtype = T::TYPE_STR
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    while buffer.len() > output_size {
        let output = Buffer::builder()
            .queue(queue.clone())
            .len(buffer.len() / stride)
            .build()?;

        let wg_size = if stride < WG_SIZE {
            stride
        } else {
            debug_assert_eq!(stride % WG_SIZE, 0);
            WG_SIZE
        };

        let kernel = Kernel::builder()
            .name("reduce_axis")
            .program(&program)
            .queue(queue.clone())
            .local_work_size(wg_size)
            .global_work_size(buffer.len())
            .arg(init)
            .arg(&buffer)
            .arg(&output)
            .arg_local::<T>(wg_size)
            .build()?;

        unsafe { kernel.enq()? }

        buffer = output;
        stride /= wg_size;
        debug_assert_eq!(output_size * stride, buffer.len());
    }

    Ok(buffer)
}

fn fold_axis<T: CDatatype>(
    init: T,
    reduce: &'static str,
    queue: Queue,
    input: &Buffer<T>,
    reduce_dim: usize,
    target_dim: usize,
) -> Result<Buffer<T>, Error> {
    debug_assert_eq!(input.len() % reduce_dim, 0);

    let output_size = (input.len() / reduce_dim) * target_dim;

    let src = format!(
        r#"
        inline void add({dtype}* left, const {dtype} right) {{
            *left += right;
        }}

        inline void mul({dtype}* left, const {dtype} right) {{
            *left *= right;
        }}

        __kernel void fold_axis(
            const ulong reduce_dim,
            const ulong target_dim,
            {dtype} init,
            __global const {dtype}* input,
            __global {dtype}* output)
        {{
            // the global offset in the output basis
            const ulong o_offset = get_global_id(0);

            // the local coordinate in the outer dimension
            const ulong a = o_offset / target_dim;

            // the local coordinate in the dimension to reduce
            const ulong b = o_offset % target_dim;

            // the global offset in the input basis
            const ulong i_offset = (a * reduce_dim) + b;

            {dtype} reduced = init;

            for (uint stride = i_offset; stride < (a + 1) * reduce_dim; stride += target_dim) {{
                {reduce}(&reduced, input[stride]);
            }}

            output[o_offset] = reduced;
        }}
        "#,
        dtype = T::TYPE_STR,
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(output_size)
        .build()?;

    let kernel = Kernel::builder()
        .name("fold_axis")
        .program(&program)
        .queue(queue.clone())
        .global_work_size(output_size)
        .arg(reduce_dim as u64)
        .arg(target_dim as u64)
        .arg(init)
        .arg(input)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}
