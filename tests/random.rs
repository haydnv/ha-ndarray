use ha_ndarray::{ArrayBase, Error, NDArrayCompareScalar, NDArrayReduce};

#[test]
fn test_random_normal() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayBase::random_normal(vec![size], None)?;

    assert!(!array.eq(0.)?.any()?);
    assert_eq!(array.sum()? as usize / size, 0);
    assert!(array.gt(1.)?.any()?);
    assert!(array.lt(-1.)?.any()?);

    Ok(())
}

#[test]
fn test_random_uniform() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayBase::random_uniform(vec![1_000_000], None)?;

    assert!(!array.eq(0.)?.any()?);
    assert_eq!(array.sum()? as usize / size, 0);
    assert!(array.gte(-1.)?.all()?);
    assert!(array.lte(1.)?.all()?);

    Ok(())
}
