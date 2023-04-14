use ha_ndarray::{ArrayBase, Error, NDArrayCompare, NDArrayReduce};

#[test]
fn test_add() -> Result<(), Error> {
    let shape = vec![5, 2];
    let left = ArrayBase::from_vec(shape.to_vec(), (0..10).into_iter().collect())?;
    let right = ArrayBase::from_vec(shape.to_vec(), (0..10).into_iter().rev().collect())?;
    let actual = left + right;
    let expected = ArrayBase::constant(shape, 9);
    assert!(expected.eq(&actual)?.all()?);
    Ok(())
}
