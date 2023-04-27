use ha_ndarray::construct::{RandomNormal, RandomUniform};
use ha_ndarray::*;

const LEARNING_RATE: f32 = 0.0001;
const NUM_EXAMPLES: usize = 25;

fn main() -> Result<(), Error> {
    let weights = RandomNormal::new(2)?;
    let weights = (ArrayOp::new(vec![2, 1], weights) - 0.5) * 2.;
    let weights = ArrayBase::copy(&weights)?;

    let inputs = RandomUniform::new(NUM_EXAMPLES * 2)?;
    let inputs = ArrayOp::new(vec![NUM_EXAMPLES, 2], inputs) * 2.;
    let inputs = ArrayBase::copy(&inputs)?;

    let inputs_bool = inputs.lt_scalar(1.0);
    let labels = ArrayBase::copy(
        &inputs_bool
            .slice(vec![(0..NUM_EXAMPLES).into(), 0.into()])?
            .and(&(inputs_bool.slice(vec![(0..NUM_EXAMPLES).into(), 1.into()])?))?
            .expand_dim(1)?
            .cast(),
    )?;

    loop {
        let output = inputs.matmul(&weights)?;
        let error = labels.clone() - output;
        let loss = error.pow(2.);
        println!("loss: {} (max {})", loss.sum()?, loss.max()?);
        assert!(!loss.is_inf().any()?, "diverged");

        if loss.lt_scalar(1.0).all()? {
            return Ok(());
        } else {
            let d_loss = error * 2.;
            let weights_t = weights.transpose(None)?;
            let gradient = d_loss.matmul(&weights_t)?;
            let deltas = gradient.sum_axis(0)?.expand_dim(1)?;

            weights.write(&(weights.clone() + (deltas * LEARNING_RATE)))?;
            assert!(!weights.is_nan().any()?);
        }
    }
}
