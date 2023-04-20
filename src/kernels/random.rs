use ocl::{Buffer, Error, Kernel, Program, Queue};

use super::{div_ceil, WG_SIZE};

const LIB: &'static str = r#"
const float pi = 3.14159;
const float resolution = 1.0 / ((float) UINT_MAX);

// PCG hash by Melissa E. O'Neill: https://www.pcg-random.org/
uint pcg_hash(uint seed) {
    uint state = seed * 747796405 + 2891336453;
    uint word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    return (word >> 22) ^ word;
}

// Xorshift algorithm by George Marsalia: https://www.jstatsoft.org/article/view/v008i14
uint xorshift(uint rng_state) {
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

float random(const ulong seed, const ulong offset) {
    // rotate the offset seed number of places through the range 0..2^32
    uint rng_state = (offset + seed) & UINT_MAX;

    rng_state = xorshift(rng_state);

    for (uint i = 32; i < log2((float) offset); i++) {{
        rng_state = xorshift(rng_state);
    }}

    rng_state = pcg_hash(rng_state);

    return rng_state * resolution;
}
"#;

pub fn random_normal(queue: Queue, seed: usize, size: usize) -> Result<Buffer<f32>, Error> {
    assert!(seed as u64 <= u64::MAX - u32::MAX as u64);

    let src = format!(
        r#"
        {LIB}

        __kernel void random_normal(
                const ulong seed,
                __global float* output,
                __local float* normal)
        {{
            const ulong global_offset = get_global_id(0);
            const uint local_offset = get_local_id(0);

            normal[local_offset] = random(seed, global_offset);
            barrier(CLK_LOCAL_MEM_FENCE);

            // Box-Muller algorithm
            if (local_offset % 2 == 0) {{
                float u1 = normal[local_offset];
                float u2 = normal[local_offset + 1];
                float r = sqrt(-2 * log(u1));
                float theta = 2 * pi * u2;
                output[global_offset] = r * cos(theta);
            }} else {{
                float u1 = normal[local_offset - 1];
                float u2 = normal[local_offset];
                float r = sqrt(-2 * log(u1));
                float theta = 2 * pi * u2;
                output[global_offset] = r * sin(theta);
            }}
        }}
    "#
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder()
        .queue(queue.clone())
        .len(WG_SIZE * div_ceil(size, WG_SIZE))
        .build()?;

    let kernel = Kernel::builder()
        .name("random_normal")
        .queue(queue)
        .program(&program)
        .global_work_size(output.len())
        .local_work_size(WG_SIZE)
        .arg(u64::try_from(seed).expect("seed"))
        .arg(&output)
        .arg_local::<f32>(WG_SIZE)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}

pub fn random_uniform(queue: Queue, seed: usize, size: usize) -> Result<Buffer<f32>, Error> {
    assert!(seed as u64 <= u64::MAX - u32::MAX as u64);

    let src = format!(
        r#"
        {LIB}

        __kernel void random_uniform(const ulong seed, __global float* output) {{
            const ulong offset = get_global_id(0);
            output[offset] = random(seed, offset) * 2. - 1.;
        }}
    "#
    );

    let program = Program::builder().source(src).build(&queue.context())?;

    let output = Buffer::builder().queue(queue.clone()).len(size).build()?;

    let kernel = Kernel::builder()
        .name("random_uniform")
        .queue(queue)
        .program(&program)
        .global_work_size(output.len())
        .arg(u64::try_from(seed).expect("seed"))
        .arg(&output)
        .build()?;

    unsafe { kernel.enq()? }

    Ok(output)
}
