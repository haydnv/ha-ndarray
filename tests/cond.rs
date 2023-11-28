use ha_ndarray::*;

#[test]
fn test_cond() -> Result<(), Error> {
    let size = 10;

    let data = (0..size)
        .into_iter()
        .map(|n| if n % 2 == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();

    let cond = ArrayBuf::new(data, shape![size])?;

    let left = ArrayBuf::constant(1, shape![size])?;
    let right = ArrayBuf::constant(0, shape![size])?;

    let result = cond.as_ref::<[u8]>().cond(left, right)?;

    // let cond = cond.cast::<f32>()?;
    let eq = result.eq(cond.as_ref::<[u8]>())?;
    assert!(eq.all()?);

    Ok(())
}
