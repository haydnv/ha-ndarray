use ha_ndarray::*;

#[test]
fn test_range() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayOp::range(0f32, 500_000f32, size)?;
    let buffer = array.into_inner();

    for (i, a) in buffer.read()?.to_slice()?.into_iter().copied().enumerate() {
        let e = (i as f32) / 2.0;
        assert_eq!(e, a);
    }

    Ok(())
}

#[test]
fn test_random_normal() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayOp::random_normal(size)?;

    // assert!(!array.as_ref().eq_scalar(0.)?.any()?);
    assert_eq!(array.as_ref().sum_all()? as usize / size, 0);
    // assert!(array.as_ref().gt_scalar(1.)?.any()?);
    // assert!(array.lt_scalar(-1.)?.any()?);

    Ok(())
}

#[test]
fn test_random_uniform() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayOp::random_uniform(size)?;

    // assert!(!array.as_ref().eq_scalar(0.)?.any()?);
    assert_eq!(array.as_ref().sum_all()? as usize / size, 0);
    // assert!(array.as_ref().ge_scalar(-1.)?.all()?);
    // assert!(array.le_scalar(1.)?.all()?);

    Ok(())
}
