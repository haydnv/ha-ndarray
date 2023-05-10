use ha_ndarray::*;
use std::sync::Arc;

#[test]
fn test_constant_array() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let array = constant(context.clone(), 0., vec![2, 3])?;
    assert!(!array.any()?);

    let array = constant(context, 1., vec![2, 3])?;
    assert!(array.all()?);

    Ok(())
}

#[test]
fn test_eq() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = constant(context.clone(), 0., vec![2, 3])?;
    let ones = constant(context, 1., vec![2, 3])?;

    assert!(zeros.clone().eq(zeros.clone())?.all()?);
    assert!(ones.clone().eq(ones.clone())?.all()?);
    assert!(!zeros.clone().eq(ones.clone())?.any()?);
    assert!(!ones.eq(zeros)?.any()?);

    Ok(())
}

#[test]
fn test_gt() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = constant(context.clone(), 0., vec![4, 5, 7])?;
    let ones = constant(context, 1., vec![4, 5, 7])?;

    assert!(!zeros.clone().gt(zeros.clone())?.any()?);
    assert!(!ones.clone().gt(ones.clone())?.any()?);
    assert!(!zeros.clone().gt(ones.clone())?.any()?);
    assert!(ones.gt(zeros)?.all()?);

    Ok(())
}

#[test]
fn test_gte() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = constant(context.clone(), 0., vec![5, 2])?;
    let ones = constant(context, 1., vec![5, 2])?;

    assert!(zeros.clone().ge(zeros.clone())?.all()?);
    assert!(ones.clone().ge(ones.clone())?.all()?);
    assert!(!zeros.clone().ge(ones.clone())?.any()?);
    assert!(ones.ge(zeros)?.all()?);

    Ok(())
}

#[test]
fn test_lt() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = constant(context.clone(), 0., vec![4, 5, 7])?;
    let ones = constant(context, 1., vec![4, 5, 7])?;

    assert!(!zeros.clone().lt(zeros.clone())?.any()?);
    assert!(!ones.clone().lt(ones.clone())?.any()?);
    assert!(zeros.clone().lt(ones.clone())?.all()?);
    assert!(!ones.lt(zeros)?.any()?);

    Ok(())
}

#[test]
fn test_lte() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = constant(context.clone(), 0., vec![5, 2])?;
    let ones = constant(context, 1., vec![5, 2])?;

    assert!(zeros.clone().le(zeros.clone())?.all()?);
    assert!(ones.clone().le(ones.clone())?.all()?);
    assert!(zeros.clone().le(ones.clone())?.all()?);
    assert!(!ones.le(zeros)?.any()?);

    Ok(())
}

#[test]
fn test_ne() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;
    let zeros = constant(context.clone(), 0., vec![2, 3])?;
    let ones = constant(context, 1., vec![2, 3])?;

    assert!(!zeros.clone().ne(zeros.clone())?.any()?);
    assert!(!ones.clone().ne(ones.clone())?.any()?);
    assert!(zeros.clone().ne(ones.clone())?.all()?);
    assert!(ones.ne(zeros)?.all()?);

    Ok(())
}

fn constant<T: CDatatype>(
    context: Context,
    value: T,
    shape: Vec<usize>,
) -> Result<ArrayBase<Arc<Vec<T>>>, Error> {
    let size = shape.iter().product();
    ArrayBase::<Arc<Vec<_>>>::with_context(context, shape, Arc::new(vec![value; size]))
}
