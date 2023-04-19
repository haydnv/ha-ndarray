use std::fmt;

use ocl::{Buffer, Error, Kernel, Program, Queue};

use crate::{AxisBound, CDatatype};

use super::ArrayFormat;

struct Bounds<'a> {
    axes: &'a [AxisBound],
}

impl<'a> Bounds<'a> {
    fn max_indices(&self) -> usize {
        self.axes
            .iter()
            .map(|bound| match bound {
                AxisBound::At(_) => 1,
                AxisBound::In(_, _, _) => 0,
                AxisBound::Of(indices) => indices.len(),
            })
            .fold(1, Ord::max)
    }

    fn size(&self) -> usize {
        self.axes
            .iter()
            .map(|bound| bound.size())
            .filter(|size| *size > 0)
            .product()
    }
}

impl<'a> fmt::Display for Bounds<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("{ ")?;

        for bound in self.axes {
            f.write_str("{ ")?;

            match bound {
                AxisBound::At(i) => {
                    write!(f, ".btype=AT, ")?;
                    write!(f, "{{ .at_index={i} }}")?;
                }
                AxisBound::In(start, stop, step) => {
                    write!(f, ".btype=IN, ")?;
                    write!(f, "{{ .in_range={{ {start}, {stop}, {step}, 0 }} }}")?;
                }
                AxisBound::Of(indices) => {
                    write!(f, ".btype=OF, ")?;

                    f.write_str("{{ .of_indices={ ")?;

                    for i in indices {
                        write!(f, "{i}, ")?;
                    }

                    f.write_str("} }}")?;
                }
            }

            f.write_str(" }, ")?;
        }

        f.write_str(" }")
    }
}

pub fn slice<T: CDatatype>(
    queue: Queue,
    input: &Buffer<T>,
    shape: &[usize],
    strides: &[usize],
    axes: &[AxisBound],
    source_strides: &[usize],
) -> Result<Buffer<T>, Error> {
    let ndim = shape.len();
    debug_assert_eq!(ndim, strides.len());

    let source_ndim = axes.len();
    debug_assert_eq!(source_ndim, source_strides.len());

    let dims = ArrayFormat::from(shape);
    let strides = ArrayFormat::from(strides);

    let bounds = Bounds { axes };
    let source_strides = ArrayFormat::from(source_strides);

    let src = format!(
        r#"
        typedef enum {{ AT, IN, OF }} RangeType;

        typedef union {{
            ulong at_index;
            ulong4 in_range;
            ulong of_indices[{max_indices}];
        }} AxisRange;

        typedef struct {{
            RangeType btype;
            AxisRange range;
        }} Bound;

        const ulong dims[{ndim}] = {dims};
        const ulong strides[{ndim}] = {strides};

        const Bound bounds[{source_ndim}] = {bounds};
        const ulong source_strides[{source_ndim}] = {source_strides};

        __kernel void slice(
                __global const {dtype}* restrict input,
                __global {dtype}* restrict output)
        {{
            const ulong offset_out = get_global_id(0);

            ulong coord[{ndim}];

            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                coord[x] = (offset_out / strides[x]) % dims[x];
            }}

            ulong offset_in = 0;
            ulong x = 0;

            #pragma unroll
            for (uint source_x = 0; source_x < {source_ndim}; source_x++) {{
                ulong i = 0;

                switch (bounds[source_x].btype) {{
                    case AT:
                        i = bounds[source_x].range.at_index;

                        break;
                    case IN:
                        i = (
                            bounds[source_x].range.in_range.x +
                            (coord[x] * bounds[source_x].range.in_range.z)
                        );

                        x++;
                        break;
                    case OF:
                        i = bounds[source_x].range.of_indices[coord[x]];

                        x++;
                        break;
                }}

                offset_in += source_strides[source_x] * i;
            }}

            output[offset_out] = input[offset_in];
        }}
        "#,
        dtype = T::TYPE_STR,
        max_indices = bounds.max_indices(),
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(bounds.size())
        .build()?;

    let kernel = Kernel::builder()
        .name("slice")
        .program(&program)
        .queue(queue)
        .global_work_size(output.len())
        .arg(input)
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}
