use ha_ndarray::construct::{RandomNormal, RandomUniform};
use ha_ndarray::{ArrayOp, Error, NDArrayCompareScalar, NDArrayReduce};

#[test]
fn test_random_normal() -> Result<(), Error> {
    let size = 1_000_000;
    let op = RandomNormal::new(size)?;
    let array = ArrayOp::new(vec![size], op);

    assert!(!array.eq_scalar(0.)?.any()?);
    assert_eq!(array.sum()? as usize / size, 0);
    assert!(array.gt_scalar(1.)?.any()?);
    assert!(array.lt_scalar(-1.)?.any()?);

    Ok(())
}

#[test]
fn test_random_uniform() -> Result<(), Error> {
    let size = 1_000_000;
    let op = RandomUniform::new(size)?;
    let array = ArrayOp::new(vec![size], op);

    assert!(!array.eq_scalar(0.)?.any()?);
    assert_eq!(array.sum()? as usize / size, 0);
    assert!(array.ge_scalar(-1.)?.all()?);
    assert!(array.le_scalar(1.)?.all()?);

    Ok(())
}
