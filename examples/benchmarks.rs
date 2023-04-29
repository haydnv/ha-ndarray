use std::time::Instant;

use ha_ndarray::{
    ArrayBase, Context, Error, NDArray, NDArrayMath, NDArrayRead, NDArrayReduce, NDArrayTransform,
};

const ITERATIONS: usize = 20;

fn broadcast_and_multiply(context: Context) -> Result<(), Error> {
    let shape = vec![40, 10, 20, 30];
    let queue = context.queue(shape.iter().product())?;

    let left = ArrayBase::with_context(context.clone(), vec![10, 20, 30], vec![1.0f64; 6000])?;
    let right = ArrayBase::with_context(context, vec![40, 10, 1, 30], vec![1.0f64; 12000])?;

    println!("broadcast and multiply {:?} and {:?}...", left, right);
    let product = NDArrayMath::mul(&left.broadcast(shape.to_vec())?, &right.broadcast(shape)?)?;

    for _ in 0..ITERATIONS {
        let start = Instant::now();
        product.read(&queue)?;
        let duration = start.elapsed();
        println!("{:?} us", duration.as_micros());
    }

    Ok(())
}

fn reduce_sum(context: Context) -> Result<(), Error> {
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
    let context = Context::default()?;

    broadcast_and_multiply(context.clone())?;
    reduce_sum(context.clone())?;
    transpose(context)?;

    Ok(())
}
