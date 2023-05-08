use ha_ndarray::*;

#[test]
fn test_add() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let shape = vec![5, 2];

    // TODO: use a range constructor
    let left = ArrayBase::<Vec<_>>::with_context(
        context.clone(),
        shape.to_vec(),
        (0..10).into_iter().collect(),
    )?;

    // TODO: use a range constructor
    let right = ArrayBase::<Vec<_>>::with_context(
        context,
        shape.to_vec(),
        (0..10).into_iter().rev().collect(),
    )?;

    let actual = left + right;
    let expected = ArrayBase::new(shape, vec![9; 10])?;
    assert!(expected.eq(actual)?.all()?);
    Ok(())
}

#[test]
fn test_expand_and_broadcast_and_sub() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;

    // TODO: use a range constructor
    let left = ArrayBase::<Vec<_>>::with_context(
        context.clone(),
        vec![2, 3],
        (0i32..6).into_iter().collect(),
    )?;

    let right = ArrayBase::with_context(context.clone(), vec![2], vec![0, 1])?;

    let expected = ArrayBase::new(vec![2, 3], vec![0, 1, 2, 2, 3, 4])?;
    let actual = left - right.expand_dims(vec![1])?.broadcast(vec![2, 3])?;
    assert!(expected.eq(actual)?.all()?);
    Ok(())
}
