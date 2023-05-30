use std::fmt;

use ocl::{Error, Program};

use crate::{AxisBound, CDatatype, Context};

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

pub fn read_slice<T: CDatatype>(
    context: &Context,
    shape: &[usize],
    strides: &[usize],
    axes: &[AxisBound],
    source_strides: &[usize],
) -> Result<Program, Error> {
    let ndim = shape.len();
    assert_eq!(ndim, strides.len());

    let source_ndim = axes.len();
    assert_eq!(source_ndim, source_strides.len());

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

        __kernel void read_slice(
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

    Program::builder().source(src).build(context.cl_context())
}

pub fn write_to_slice<T: CDatatype>(
    context: &Context,
    shape: &[usize],
    strides: &[usize],
    axes: &[AxisBound],
    source_strides: &[usize],
) -> Result<Program, Error> {
    let ndim = shape.len();
    assert_eq!(ndim, strides.len());

    let source_ndim = axes.len();
    assert_eq!(source_ndim, source_strides.len());

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

        __kernel void write_slice(
                __global {dtype}* restrict output,
                __global const {dtype}* restrict input)
        {{
            const ulong offset_in = get_global_id(0);

            ulong coord[{ndim}];

            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                coord[x] = (offset_in / strides[x]) % dims[x];
            }}

            ulong offset_out = 0;
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

                offset_out += source_strides[source_x] * i;
            }}

            output[offset_out] = input[offset_in];
        }}
        "#,
        dtype = T::TYPE_STR,
        max_indices = bounds.max_indices(),
    );

    Program::builder().source(src).build(context.cl_context())
}

pub fn write_value_to_slice<T: CDatatype>(
    context: &Context,
    shape: &[usize],
    strides: &[usize],
    axes: &[AxisBound],
    source_strides: &[usize],
) -> Result<Program, Error> {
    let ndim = shape.len();
    assert_eq!(ndim, strides.len());

    let source_ndim = axes.len();
    assert_eq!(source_ndim, source_strides.len());

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

        __kernel void write_slice_value(
                __global {dtype}* restrict output,
                const {dtype} input)
        {{
            const ulong offset_in = get_global_id(0);

            ulong coord[{ndim}];

            #pragma unroll
            for (uint x = 0; x < {ndim}; x++) {{
                coord[x] = (offset_in / strides[x]) % dims[x];
            }}

            ulong offset_out = 0;
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

                offset_out += source_strides[source_x] * i;
            }}

            output[offset_out] = input;
        }}
        "#,
        dtype = T::TYPE_STR,
        max_indices = bounds.max_indices(),
    );

    Program::builder().source(src).build(context.cl_context())
}
