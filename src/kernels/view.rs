use ocl::{Buffer, Error, Kernel, Program, Queue};

use crate::CDatatype;

use super::ArrayFormat;

pub fn reorder_inplace<T: CDatatype>(
    queue: Queue,
    buffer: Buffer<T>,
    shape: &[usize],
    strides: &[usize],
    source_strides: &[usize],
) -> Result<Buffer<T>, Error> {
    let ndim = shape.len();
    debug_assert_eq!(strides.len(), ndim);

    let source_ndim = source_strides.len();

    let dims = ArrayFormat::from(shape);
    let strides = ArrayFormat::from(strides);
    let source_strides = ArrayFormat::from(source_strides);

    let src = format!(
        r#"
        const uint ndim = {ndim};
        const ulong dims[{ndim}] = {dims};
        const ulong strides[{ndim}] = {strides};

        const ulong source_strides[{source_ndim}] = {source_strides};

        __kernel void reorder_inplace(__global {dtype}* restrict input) {{
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

    let kernel = Kernel::builder()
        .name("reorder_inplace")
        .program(&program)
        .queue(queue)
        .global_work_size(buffer.len())
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
    let ndim = shape.len();
    debug_assert_eq!(strides.len(), ndim);

    let source_ndim = source_strides.len();

    let dims = ArrayFormat::from(shape);
    let strides = ArrayFormat::from(strides);
    let source_strides = ArrayFormat::from(source_strides);

    let src = format!(
        r#"
        const uint ndim = {ndim};
        const ulong dims[{ndim}] = {dims};
        const ulong strides[{ndim}] = {strides};

        const ulong source_strides[{source_ndim}] = {source_strides};

        __kernel void reorder(
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

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(shape.iter().product::<usize>())
        .build()?;

    let kernel = Kernel::builder()
        .name("reorder")
        .program(&program)
        .queue(queue)
        .global_work_size(output.len())
        .arg(&input)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}
