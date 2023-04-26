use ha_ndarray::{ArrayBase, Error, NDArrayCompare, NDArrayReduce};

#[test]
fn test_constant_array() -> Result<(), Error> {
    let array = ArrayBase::constant(vec![2, 3], 0.);
    assert!(!array.any()?);

    let array = ArrayBase::constant(vec![2, 3], 1.);
    assert!(array.all()?);

    Ok(())
}

#[test]
fn test_eq() -> Result<(), Error> {
    let zeros = ArrayBase::constant(vec![2, 3], 0.);
    let ones = ArrayBase::constant(vec![2, 3], 1.);

    assert!(zeros.eq(&zeros)?.all()?);
    assert!(ones.eq(&ones)?.all()?);
    assert!(!zeros.eq(&ones)?.any()?);
    assert!(!ones.eq(&zeros)?.any()?);

    Ok(())
}

#[test]
fn test_gt() -> Result<(), Error> {
    let zeros = ArrayBase::constant(vec![4, 5, 7], 0.);
    let ones = ArrayBase::constant(vec![4, 5, 7], 1.);

    assert!(!zeros.gt(&zeros)?.any()?);
    assert!(!ones.gt(&ones)?.any()?);
    assert!(!zeros.gt(&ones)?.any()?);
    assert!(ones.gt(&zeros)?.all()?);

    Ok(())
}

#[test]
fn test_gte() -> Result<(), Error> {
    let zeros = ArrayBase::constant(vec![5, 2], 0.);
    let ones = ArrayBase::constant(vec![5, 2], 1.);

    assert!(zeros.ge(&zeros)?.all()?);
    assert!(ones.ge(&ones)?.all()?);
    assert!(!zeros.ge(&ones)?.any()?);
    assert!(ones.ge(&zeros)?.all()?);

    Ok(())
}

#[test]
fn test_lt() -> Result<(), Error> {
    let zeros = ArrayBase::constant(vec![4, 5, 7], 0.);
    let ones = ArrayBase::constant(vec![4, 5, 7], 1.);

    assert!(!zeros.lt(&zeros)?.any()?);
    assert!(!ones.lt(&ones)?.any()?);
    assert!(zeros.lt(&ones)?.all()?);
    assert!(!ones.lt(&zeros)?.any()?);

    Ok(())
}

#[test]
fn test_lte() -> Result<(), Error> {
    let zeros = ArrayBase::constant(vec![5, 2], 0.);
    let ones = ArrayBase::constant(vec![5, 2], 1.);

    assert!(zeros.le(&zeros)?.all()?);
    assert!(ones.le(&ones)?.all()?);
    assert!(zeros.le(&ones)?.all()?);
    assert!(!ones.le(&zeros)?.any()?);

    Ok(())
}

#[test]
fn test_ne() -> Result<(), Error> {
    let zeros = ArrayBase::constant(vec![2, 3], 0.);
    let ones = ArrayBase::constant(vec![2, 3], 1.);

    assert!(!zeros.ne(&zeros)?.any()?);
    assert!(!ones.ne(&ones)?.any()?);
    assert!(zeros.ne(&ones)?.all()?);
    assert!(ones.ne(&zeros)?.all()?);

    Ok(())
}
