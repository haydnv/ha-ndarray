use ha_ndarray::*;

const LEARNING_RATE: f32 = 0.0001;
const NUM_EXAMPLES: usize = 25;

fn main() -> Result<(), Error> {
    let weights = ArrayBase::random_normal(vec![2, 1], None)?;

    let inputs = (ArrayBase::random_uniform(vec![NUM_EXAMPLES, 2], None)? + 1.).copy()?;
    let inputs_bool = inputs.lt_scalar(1.0);
    let labels = inputs_bool
        .slice(vec![(0..NUM_EXAMPLES).into(), 0.into()])?
        .and(&(inputs_bool.slice(vec![(0..NUM_EXAMPLES).into(), 1.into()])?))?
        .expand_dim(1)?
        .cast()
        .copy()?;

    loop {
        let output = inputs.matmul(&weights)?.copy()?;

        let loss = (labels.clone() - output.clone()).pow(2.);
        println!("loss: {} (max {})", loss.sum()?, loss.max()?);

        if loss.lt_scalar(1.0).all()? {
            return Ok(());
        } else {
            let d_loss = (labels.clone() - output) * 2.;
            let weights_t = weights.transpose(None)?;
            let gradient = d_loss.matmul(&weights_t)?;
            let deltas = gradient.sum_axis(0)?.expand_dim(1)?;

            weights.write(&(weights.clone() + (deltas * LEARNING_RATE)))?;
            assert!(!weights.is_nan().any()?);
        }
    }
}
