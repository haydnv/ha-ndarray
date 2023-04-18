use ocl::{Buffer, Error, Kernel, Program, Queue};

use crate::CDatatype;

pub fn reorder_inplace<T: CDatatype>(
    queue: Queue,
    buffer: Buffer<T>,
    shape: &[usize],
    strides: &[usize],
    source_strides: &[usize],
) -> Result<Buffer<T>, Error> {
    let src = format!(
        r#"
        __kernel void reorder(
                const uint ndim,
                __global const ulong* restrict dims,
                __global const ulong* restrict strides,
                __global const ulong* restrict source_strides,
                __global {dtype}* restrict input)
        {{
            ulong offset = get_global_id(0);

            ulong coord[{ndim}];
            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                if (strides[x] == 0) {{
                    coord[x] = 0;
                }} else {{
                    coord[x] = (offset / strides[x]) % dims[x];
                }}
            }}

            ulong source_offset = 0;
            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                source_offset += coord[x] * source_strides[x];
            }}

            const {dtype} n = input[source_offset];
            barrier(CLK_LOCAL_MEM_FENCE);
            input[offset] = source_offset;
        }}
        "#,
        dtype = T::TYPE_STR,
        ndim = shape.len(),
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let (dims, strides, source_strides) =
        build_args(queue.clone(), shape, strides, source_strides)?;

    let kernel = Kernel::builder()
        .name("reorder")
        .program(&program)
        .queue(queue)
        .global_work_size(buffer.len())
        .arg(u32::try_from(shape.len()).expect("ndim"))
        .arg(&dims)
        .arg(&strides)
        .arg(&source_strides)
        .arg(&buffer)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(buffer)
}

pub fn reorder<T: CDatatype>(
    queue: Queue,
    input: Buffer<T>,
    shape: &[usize],
    strides: &[usize],
    source_strides: &[usize],
) -> Result<Buffer<T>, Error> {
    let src = format!(
        r#"
        __kernel void reorder(
                const uint ndim,
                __global const ulong* restrict dims,
                __global const ulong* restrict strides,
                __global const ulong* restrict source_strides,
                __global const {dtype}* restrict input,
                __global {dtype}* restrict output)
        {{
            ulong offset = get_global_id(0);

            ulong coord[{ndim}];
            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                if (strides[x] == 0) {{
                    coord[x] = 0;
                }} else {{
                    coord[x] = (offset / strides[x]) % dims[x];
                }}
            }}

            ulong source_offset = 0;
            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                source_offset += coord[x] * source_strides[x];
            }}

            output[offset] = input[source_offset];
        }}
        "#,
        dtype = T::TYPE_STR,
        ndim = shape.len(),
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let (dims, strides, source_strides) =
        build_args(queue.clone(), shape, strides, source_strides)?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(shape.iter().product::<usize>())
        .build()?;

    let kernel = Kernel::builder()
        .name("reorder")
        .program(&program)
        .queue(queue)
        .global_work_size(output.len())
        .arg(u32::try_from(shape.len()).expect("ndim"))
        .arg(&dims)
        .arg(&strides)
        .arg(&source_strides)
        .arg(&input)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}

pub fn build_args(
    queue: Queue,
    shape: &[usize],
    strides: &[usize],
    source_strides: &[usize],
) -> Result<(Buffer<usize>, Buffer<usize>, Buffer<usize>), Error> {
    debug_assert_eq!(shape.len(), strides.len());
    debug_assert_eq!(shape.len(), source_strides.len());

    let dims = Buffer::builder()
        .queue(queue.clone())
        .copy_host_slice(shape)
        .len(shape.len())
        .build()?;

    let strides = Buffer::builder()
        .queue(queue.clone())
        .copy_host_slice(strides)
        .len(strides.len())
        .build()?;

    let source_strides = Buffer::builder()
        .queue(queue.clone())
        .copy_host_slice(source_strides)
        .len(strides.len())
        .build()?;

    Ok((dims, strides, source_strides))
}
