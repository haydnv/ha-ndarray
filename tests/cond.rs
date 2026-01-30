use ha_ndarray::*;

#[test]
fn test_cond() -> Result<(), Error> {
    let size = 2048;

    let cond = (0..size)
        .map(|n| if n % 2 == 0 { 1u8 } else { 0u8 })
        .collect::<Vec<_>>();

    let then = ArrayBuf::constant(1u8, shape![size])?;
    let or_else = ArrayBuf::constant(0u8, shape![size])?;

    let actual = ArrayBuf::new(Buffer::from(cond.clone()), shape![size])?.cond(then, or_else)?;

    assert_eq!(actual.buffer()?.to_slice()?.into_vec(), cond);

    Ok(())
}
