use ha_ndarray::construct::{RandomNormal, RandomUniform, Range};
use ha_ndarray::*;

#[test]
fn test_range() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let size = 1_000_000;
    let op = Range::with_context(context, 0f32, 500_000f32, size)?;
    let array = ArrayOp::new(vec![size], op);
    let array = ArrayBase::<Vec<f32>>::copy(&array)?;
    let buffer = array.into_inner();

    for (i, a) in buffer.into_iter().enumerate() {
        let e = (i as f32) / 2.0;
        assert_eq!(e, a);
    }

    Ok(())
}

#[test]
fn test_random_normal() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let size = 1_000_000;
    let op = RandomNormal::with_context(context, size)?;
    let array = ArrayOp::new(vec![size], op);

    assert!(!array.clone().eq_scalar(0.)?.any()?);
    assert_eq!(array.clone().sum()? as usize / size, 0);
    assert!(array.clone().gt_scalar(1.)?.any()?);
    assert!(array.lt_scalar(-1.)?.any()?);

    Ok(())
}

#[test]
fn test_random_uniform() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let size = 1_000_000;
    let op = RandomUniform::with_context(context, size)?;
    let array = ArrayOp::new(vec![size], op);

    assert!(!array.clone().eq_scalar(0.)?.any()?);
    assert_eq!(array.clone().sum()? as usize / size, 0);
    assert!(array.clone().ge_scalar(-1.)?.all()?);
    assert!(array.clone().le_scalar(1.)?.all()?);

    Ok(())
}
