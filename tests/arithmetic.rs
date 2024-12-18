use ha_ndarray::*;

#[test]
fn test_add() -> Result<(), Error> {
    let shape = shape![5, 2];

    let left = ArrayOp::range(0, 10, shape.clone())?;
    let right = ArrayBuf::new((0..10).into_iter().rev().collect::<Vec<_>>(), shape.clone())?;

    let actual = left.add(right)?;
    let expected = ArrayBuf::constant(9, shape)?;
    assert!(expected.eq(actual)?.all()?);
    Ok(())
}

#[test]
fn test_expand_and_broadcast_and_sub() -> Result<(), Error> {
    let left = ArrayOp::range(0, 6, shape![2, 3])?;
    let right = ArrayBuf::new(vec![0, 1], shape![2])?;

    let expected = ArrayBuf::new(vec![0, 1, 2, 2, 3, 4], shape![2, 3])?;
    let actual = left.sub(right.unsqueeze(axes![1])?.broadcast(shape![2, 3])?)?;
    assert!(expected.eq(actual)?.all()?);
    Ok(())
}
