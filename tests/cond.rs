use ha_ndarray::*;

#[test]
fn test_cond() -> Result<(), Error> {
    let size = 2048;

    let data = (0..size)
        .into_iter()
        .map(|n| if n % 2 == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();

    let cond = ArrayBuf::new(data, shape![size])?;

    let then = ArrayBuf::constant(1., shape![size])?;
    let or_else = ArrayBuf::constant(0., shape![size])?;

    let actual = cond.as_ref::<[u8]>().cond(then, or_else)?;

    let cond: Array<f32, _> = cond.as_ref::<[u8]>().cast()?;
    let eq = actual.eq(cond)?;
    assert!(eq.all()?);

    Ok(())
}
