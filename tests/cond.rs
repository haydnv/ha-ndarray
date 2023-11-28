use ha_ndarray::*;

#[test]
fn test_cond() -> Result<(), Error> {
    let size = 10;

    let data = (0..size)
        .into_iter()
        .map(|n| if n % 2 == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();

    let cond = ArrayBuf::new(data, shape![size])?;

    let then = ArrayBuf::constant(1, shape![size])?;
    let or_else = ArrayBuf::constant(0, shape![size])?;

    let actual = cond.as_ref::<[u8]>().cond(then, or_else)?;

    // let cond = cond.cast::<f32>()?;
    let eq = actual.eq(cond.as_ref::<[u8]>())?;
    assert!(eq.all()?);

    Ok(())
}
