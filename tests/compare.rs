use ha_ndarray::*;

#[test]
fn test_constant_array() -> Result<(), Error> {
    let array = ArrayBuf::constant(0, shape![2, 3])?;
    assert!(!array.any()?);

    let array = ArrayBuf::constant(1., shape![2, 3])?;
    assert!(array.all()?);

    Ok(())
}

#[test]
fn test_eq() -> Result<(), Error> {
    let zeros = ArrayBuf::constant(0., shape![2, 3])?;
    let ones = ArrayBuf::constant(1., shape![2, 3])?;

    assert!(zeros.as_ref().eq(zeros.as_ref())?.all()?);
    assert!(ones.as_ref().eq(ones.as_ref())?.all()?);
    assert!(!zeros.as_ref().eq(ones.as_ref())?.any()?);
    assert!(!ones.eq(zeros)?.any()?);

    Ok(())
}

#[test]
fn test_gt() -> Result<(), Error> {
    let zeros = ArrayBuf::constant(0., shape![4, 5, 7])?;
    let ones = ArrayBuf::constant(1., shape![4, 5, 7])?;

    assert!(!zeros.as_ref().gt(zeros.as_ref())?.any()?);
    assert!(!ones.as_ref().gt(ones.as_ref())?.any()?);
    assert!(!zeros.as_ref().gt(ones.as_ref())?.any()?);
    assert!(ones.gt(zeros)?.all()?);

    Ok(())
}

#[test]
fn test_gte() -> Result<(), Error> {
    let zeros = ArrayBuf::constant(0., shape![5, 2])?;
    let ones = ArrayBuf::constant(1., shape![5, 2])?;

    assert!(zeros.as_ref().ge(zeros.as_ref())?.all()?);
    assert!(ones.as_ref().ge(ones.as_ref())?.all()?);
    assert!(!zeros.as_ref().ge(ones.as_ref())?.any()?);
    assert!(ones.ge(zeros)?.all()?);

    Ok(())
}

#[test]
fn test_lt() -> Result<(), Error> {
    let zeros = ArrayBuf::constant(0., shape![4, 5, 7])?;
    let ones = ArrayBuf::constant(1., shape![4, 5, 7])?;

    assert!(!zeros.as_ref().lt(zeros.as_ref())?.any()?);
    assert!(!ones.as_ref().lt(ones.as_ref())?.any()?);
    assert!(zeros.as_ref().lt(ones.as_ref())?.all()?);
    assert!(!ones.lt(zeros)?.any()?);

    Ok(())
}

#[test]
fn test_lte() -> Result<(), Error> {
    let zeros = ArrayBuf::constant(0., shape![5, 2])?;
    let ones = ArrayBuf::constant(1., shape![5, 2])?;

    assert!(zeros.as_ref().le(zeros.as_ref())?.all()?);
    assert!(ones.as_ref().le(ones.as_ref())?.all()?);
    assert!(zeros.as_ref().le(ones.as_ref())?.all()?);
    assert!(!ones.le(zeros)?.any()?);

    Ok(())
}

#[test]
fn test_ne() -> Result<(), Error> {
    let zeros = ArrayBuf::constant(0., shape![2, 3])?;
    let ones = ArrayBuf::constant(1., shape![2, 3])?;

    assert!(!zeros.as_ref().ne(zeros.as_ref())?.any()?);
    assert!(!ones.as_ref().ne(ones.as_ref())?.any()?);
    assert!(zeros.as_ref().ne(ones.as_ref())?.all()?);
    assert!(ones.ne(zeros)?.all()?);

    Ok(())
}
