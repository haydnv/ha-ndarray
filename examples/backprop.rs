use std::sync::{Arc, RwLock};

use ha_ndarray::construct::{RandomNormal, RandomUniform};
use ha_ndarray::*;

const LEARNING_RATE: f32 = 0.0001;
const NUM_EXAMPLES: usize = 2048;

fn main() -> Result<(), Error> {
    let context = Context::default()?;
    let weights = RandomNormal::with_context(context.clone(), 2)?;
    let weights = ArrayOp::new(vec![2, 1], weights) - 0.5;
    let mut weights = ArrayBase::<Arc<RwLock<Buffer<f32>>>>::copy(&weights)?;

    let inputs = RandomUniform::with_context(context, vec![NUM_EXAMPLES, 2])?;
    let inputs = ArrayOp::new(vec![NUM_EXAMPLES, 2], inputs) * 2.;
    let inputs = ArrayBase::<Arc<Buffer<f32>>>::copy(&inputs)?;

    let inputs_bool = inputs.clone().lt_scalar(1.0)?;

    let inputs_left = inputs_bool
        .clone()
        .slice(vec![(0..NUM_EXAMPLES).into(), 0.into()])?;

    let inputs_right = inputs_bool.slice(vec![(0..NUM_EXAMPLES).into(), 1.into()])?;

    let labels = inputs_left
        .and(inputs_right)?
        .expand_dims(vec![1])?
        .cast()?;

    let labels = ArrayBase::<Buffer<f32>>::copy(&labels)?;

    let output = inputs.matmul(weights.clone())?;
    let error = labels.sub(output)?;
    let loss = error.clone().pow_scalar(2.)?;

    let d_loss = error * 2.;
    let weights_t = weights.clone().transpose(None)?;
    let gradient = d_loss.matmul(weights_t)?;
    let deltas = gradient.sum(vec![0], false)?.expand_dims(vec![1])?;
    let new_weights = weights.clone().add(deltas * LEARNING_RATE)?;

    let mut i = 0;
    loop {
        let loss = ArrayBase::<Buffer<f32>>::copy(&loss)?;

        if loss.clone().lt_scalar(1.0)?.all()? {
            return Ok(());
        }

        if i % 100 == 0 {
            println!(
                "loss: {} (max {})",
                loss.clone().sum_all()?,
                loss.clone().max_all()?
            );
        }

        assert!(!loss.clone().is_inf()?.any()?, "divergence at iteration {i}");
        assert!(!loss.is_nan()?.any()?, "unstable by iteration {i}");

        weights.write(&new_weights)?;

        i += 1;
    }
}
