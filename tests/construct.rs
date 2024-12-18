use ha_ndarray::*;

#[test]
fn test_range() -> Result<(), Error> {
    use rayon::prelude::*;

    let size = 1_000_000;

    let expected = ArrayBuf::new(
        (0..size)
            .into_par_iter()
            .map(|n| n as f32 * 0.5)
            .collect::<Vec<f32>>(),
        shape![size],
    )?;

    let actual = ArrayOp::range(0f32, 500_000f32, shape![1_000_000])?;

    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_random_normal() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayOp::random_normal(size)?;

    assert_eq!(array.as_ref().sum_all()? as usize / size, 0);
    assert!(array.as_ref().gt_scalar(1.)?.any()?);
    assert!(array.lt_scalar(-1.)?.any()?);

    Ok(())
}

#[test]
fn test_random_uniform() -> Result<(), Error> {
    let size = 1_000_000;
    let array = ArrayOp::random_uniform(size)?;

    assert_eq!(array.as_ref().sum_all()? as usize / size, 0);
    assert!(array.as_ref().ge_scalar(-1.)?.all()?);
    assert!(array.le_scalar(1.)?.all()?);

    Ok(())
}
