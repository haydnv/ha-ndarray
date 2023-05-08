use ha_ndarray::*;

#[test]
fn test_gather_cond() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let size = 10;

    let cond = ArrayBase::<Vec<_>>::with_context(
        context.clone(),
        vec![size],
        (0..size)
            .into_iter()
            .map(|n| if n % 2 == 0 { 1 } else { 0 })
            .collect(),
    )?;

    let left = ArrayBase::with_context(context.clone(), vec![size], vec![1.; size])?;
    let right = ArrayBase::with_context(context, vec![size], vec![0.; size])?;

    let result = cond.gather_cond(&left, &right)?;

    let cond = cond.cast::<f32>()?;
    let eq = result.eq(&cond)?;
    assert!(eq.all()?);

    Ok(())
}
