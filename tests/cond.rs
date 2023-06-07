use std::sync::Arc;

use ha_ndarray::*;

#[test]
fn test_cond() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let size = 10;

    let data = (0..size)
        .into_iter()
        .map(|n| if n % 2 == 0 { 1 } else { 0 })
        .collect::<Vec<_>>();

    let cond = ArrayBase::<Arc<Vec<_>>>::with_context(context.clone(), vec![size], Arc::new(data))?;

    let left = ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size], vec![1.; size])?;
    let right = ArrayBase::<Vec<_>>::with_context(context, vec![size], vec![0.; size])?;

    let result = cond.clone().cond(left, right)?;

    let cond = cond.cast::<f32>()?;
    let eq = result.eq(cond)?;
    assert!(eq.all()?);

    Ok(())
}
