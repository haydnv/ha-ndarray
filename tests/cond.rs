use ha_ndarray::*;

#[test]
fn test_cond() -> Result<(), Error> {
    let size = 2048;

    let data = (0..size)
        .into_iter()
        .map(|n| if n % 2 == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();

    let cond = ArrayBuf::new(Buffer::from(data), shape![size])?;

    let then = ArrayBuf::constant(1., shape![size])?;
    let or_else = ArrayBuf::constant(0., shape![size])?;

    let actual = cond.as_ref().cond(then, or_else)?;

    let cond: Array<f32, _> = cond.as_ref().cast()?;
    let eq = actual.eq(cond)?;
    assert!(eq.all()?);

    Ok(())
}
