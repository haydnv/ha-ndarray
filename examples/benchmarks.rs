use std::time::Instant;

use ha_ndarray::*;

const ITERATIONS: usize = 10;

fn broadcast_and_multiply(context: Context) -> Result<(), Error> {
    for m in 0..4 {
        let dim = 10usize.pow(m);
        let shape = vec![3, dim, 5, 10];
        let size = shape.iter().product::<usize>();
        let queue = context.queue(size)?;

        let left = ArrayBase::with_context(
            context.clone(),
            vec![dim, 5, 10],
            vec![1.0f64; dim * 5 * 10],
        )?;

        let right = ArrayBase::with_context(
            context.clone(),
            vec![3, dim, 1, 10],
            vec![1.0f64; 3 * dim * 10],
        )?;

        println!(
            "broadcast and multiply {:?} and {:?} (size {})...",
            left, right, size
        );

        let product = left.broadcast(shape.to_vec())? * right.broadcast(shape)?;

        for _ in 0..ITERATIONS {
            let start = Instant::now();
            product.read(&queue)?;
            let duration = start.elapsed();
            println!("{:?} us", duration.as_micros());
        }
    }

    Ok(())
}

fn matmul(context: Context) -> Result<(), Error> {
    for m in 1..16usize {
        let dim = 2usize.pow(m as u32);

        let l = ArrayBase::with_context(
            context.clone(),
            vec![16 * m, dim],
            vec![1.0f32; 16 * m * dim],
        )?;

        let r = ArrayBase::with_context(
            context.clone(),
            vec![dim, m * 32],
            vec![1.0f32; dim * m * 32],
        )?;

        let x = l.matmul(&r)?;

        let queue = context.queue(x.size())?;

        let num_ops = dim * x.size();
        println!("matmul {:?} with {:?} ({} ops)", l, r, num_ops);
        for _ in 0..ITERATIONS {
            let start = Instant::now();
            let _output = x.read(&queue)?;
            let duration = start.elapsed();
            let rate = num_ops as f32 / duration.as_secs_f32();
            println!("{:?} us @ {} M/s", duration.as_micros(), rate / 1_000_000.);
        }
    }

    Ok(())
}

fn reduce_sum_axis(context: Context) -> Result<(), Error> {
    let shape = vec![10, 20, 30, 40, 50];
    let size = shape.iter().product();
    let queue = context.queue(size)?;
    let x = ArrayBase::with_context(context, shape, vec![1; size])?;
    let reduced = x.sum_axis(2)?;

    println!("reduce {:?} (size {}) to {:?}...", x, x.size(), reduced);

    for _ in 0..ITERATIONS {
        let start = Instant::now();
        reduced.read(&queue)?;
        let duration = start.elapsed();
        println!("{:?} ms", duration.as_millis());
    }

    Ok(())
}

fn reduce_sum_all(context: Context) -> Result<(), Error> {
    let shape = vec![10, 20, 30, 40, 50];
    let size = shape.iter().product();
    let x = ArrayBase::with_context(context, shape, vec![1; size])?;

    println!("reduce {:?} (size {})...", x, x.size());

    for _ in 0..ITERATIONS {
        let start = Instant::now();
        let _x = x.sum()?;
        let duration = start.elapsed();
        println!("{:?} ms", duration.as_millis());
    }

    Ok(())
}

fn transpose(context: Context) -> Result<(), Error> {
    let shape = vec![10, 20, 30, 40, 50];
    let size = shape.iter().product();
    let queue = context.queue(size)?;
    let x = ArrayBase::with_context(context, shape, vec![1; size])?;
    let transposed = x.transpose(Some(vec![2, 4, 3, 0, 1]))?;

    println!(
        "transpose {:?} (size {}) to {:?}...",
        x,
        x.size(),
        transposed
    );

    for _ in 0..ITERATIONS {
        let start = Instant::now();
        transposed.read(&queue)?;
        let duration = start.elapsed();
        println!("{:?} ms", duration.as_millis());
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;

    broadcast_and_multiply(context.clone())?;
    matmul(context.clone())?;
    reduce_sum_axis(context.clone())?;
    reduce_sum_all(context.clone())?;
    transpose(context)?;

    Ok(())
}
