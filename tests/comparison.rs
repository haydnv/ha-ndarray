use ha_ndarray::*;

#[test]
fn test_constant_array() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let array = ArrayBase::with_context(context.clone(), vec![2, 3], vec![0.; 6])?;
    assert!(!array.any()?);

    let array = ArrayBase::with_context(context, vec![2, 3], vec![1.; 6])?;
    assert!(array.all()?);

    Ok(())
}

#[test]
fn test_eq() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = ArrayBase::with_context(context.clone(), vec![2, 3], vec![0.; 6])?;
    let ones = ArrayBase::with_context(context, vec![2, 3], vec![1.; 6])?;

    assert!(zeros.eq(&zeros)?.all()?);
    assert!(ones.eq(&ones)?.all()?);
    assert!(!zeros.eq(&ones)?.any()?);
    assert!(!ones.eq(&zeros)?.any()?);

    Ok(())
}

#[test]
fn test_gt() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = ArrayBase::with_context(context.clone(), vec![4, 5, 7], vec![0.; 140])?;
    let ones = ArrayBase::with_context(context, vec![4, 5, 7], vec![1.; 140])?;

    assert!(!zeros.gt(&zeros)?.any()?);
    assert!(!ones.gt(&ones)?.any()?);
    assert!(!zeros.gt(&ones)?.any()?);
    assert!(ones.gt(&zeros)?.all()?);

    Ok(())
}

#[test]
fn test_gte() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = ArrayBase::with_context(context.clone(), vec![5, 2], vec![0.; 10])?;
    let ones = ArrayBase::with_context(context, vec![5, 2], vec![1.; 10])?;

    assert!(zeros.ge(&zeros)?.all()?);
    assert!(ones.ge(&ones)?.all()?);
    assert!(!zeros.ge(&ones)?.any()?);
    assert!(ones.ge(&zeros)?.all()?);

    Ok(())
}

#[test]
fn test_lt() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = ArrayBase::with_context(context.clone(), vec![4, 5, 7], vec![0.; 140])?;
    let ones = ArrayBase::with_context(context, vec![4, 5, 7], vec![1.; 140])?;

    assert!(!zeros.lt(&zeros)?.any()?);
    assert!(!ones.lt(&ones)?.any()?);
    assert!(zeros.lt(&ones)?.all()?);
    assert!(!ones.lt(&zeros)?.any()?);

    Ok(())
}

#[test]
fn test_lte() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = ArrayBase::with_context(context.clone(), vec![5, 2], vec![0.; 10])?;
    let ones = ArrayBase::with_context(context, vec![5, 2], vec![1.; 10])?;

    assert!(zeros.le(&zeros)?.all()?);
    assert!(ones.le(&ones)?.all()?);
    assert!(zeros.le(&ones)?.all()?);
    assert!(!ones.le(&zeros)?.any()?);

    Ok(())
}

#[test]
fn test_ne() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = ArrayBase::with_context(context.clone(), vec![2, 3], vec![0.; 6])?;
    let ones = ArrayBase::with_context(context, vec![2, 3], vec![1.; 6])?;

    assert!(!zeros.ne(&zeros)?.any()?);
    assert!(!ones.ne(&ones)?.any()?);
    assert!(zeros.ne(&ones)?.all()?);
    assert!(ones.ne(&zeros)?.all()?);

    Ok(())
}
