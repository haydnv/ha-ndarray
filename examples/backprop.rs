use ha_ndarray::construct::{RandomNormal, RandomUniform};
use ha_ndarray::*;

const LEARNING_RATE: f32 = 0.00001;
const NUM_EXAMPLES: usize = 2048;

fn main() -> Result<(), Error> {
    let weights = RandomNormal::new(2)?;
    let weights = ArrayOp::new(vec![2, 1], weights) - 0.5;
    let weights = ArrayBase::copy(&weights)?;

    let inputs = RandomUniform::new(NUM_EXAMPLES * 2)?;
    let inputs = ArrayOp::new(vec![NUM_EXAMPLES, 2], inputs) * 2.;
    let inputs = ArrayBase::copy(&inputs)?;

    let inputs_bool = inputs.lt_scalar(1.0)?;
    let inputs_left = inputs_bool.slice(vec![(0..NUM_EXAMPLES).into(), 0.into()])?;
    let inputs_right = inputs_bool.slice(vec![(0..NUM_EXAMPLES).into(), 1.into()])?;

    let labels = ArrayBase::copy(&inputs_left.and(&inputs_right)?.expand_dim(1)?.cast()?)?;

    let output = inputs.matmul(&weights)?;
    let error = labels.sub(&output)?;
    let loss = error.pow_scalar(2.)?;

    let d_loss = error * 2.;
    let weights_t = weights.transpose(None)?;
    let gradient = d_loss.matmul(&weights_t)?;
    let deltas = gradient.sum_axis(0)?.expand_dim(1)?;
    let new_weights = weights.add(&(deltas.mul_scalar(LEARNING_RATE)?))?;

    let mut i = 0;
    loop {
        if i % 100 == 0 {
            println!("loss: {} (max {})", loss.sum()?, loss.max()?);
            assert!(!loss.is_inf()?.any()?, "diverged");
        }

        if loss.lt_scalar(1.0)?.all()? {
            return Ok(());
        }

        assert!(!weights.is_nan()?.any()?, "NaN");
        weights.write(&new_weights)?;

        i += 1;
    }
}
