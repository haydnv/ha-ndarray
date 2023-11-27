use ha_ndarray::*;

#[test]
fn test_broadcast() -> Result<(), Error> {
    let input = ArrayBuf::new(vec![1u32; 600], shape![300, 1, 2])?;
    let output = input.broadcast(shape![300, 250, 2])?;
    assert_eq!(output.shape(), &[300, 250, 2]);
    assert_eq!(output.sum_all()?, 150000);
    Ok(())
}

#[test]
fn test_slice_1d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 4, shape![4])?;

    let expected = ArrayOp::range(1, 3, shape![2])?;

    let actual = input.slice(range![(1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_slice_2d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 12, shape![4, 3])?;
    let expected = ArrayOp::range(3, 9, shape![2, 3])?;

    let actual = input.slice(range![(1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(
        expected.as_ref().eq(actual.as_ref())?.all()?,
        "expected {:?} but found {:?}",
        expected.read()?.to_slice()?,
        actual.read()?.to_slice()?
    );

    Ok(())
}

#[test]
fn test_slice_3d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 24, shape![4, 3, 2])?;

    let expected = ArrayBuf::new(vec![8, 9, 10, 11], shape![2, 2])?;

    let actual = input.slice(range![1.into(), (1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_transpose_2d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 6, shape![2, 3])?;

    let expected = ArrayBuf::new(
        vec![
            0, 3, //
            1, 4, //
            2, 5, //
        ],
        shape![3, 2],
    )?;

    let actual = input.transpose(None)?;
    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}

#[test]
fn test_transpose_3d() -> Result<(), Error> {
    let input = ArrayOp::range(0, 24, shape![2, 3, 4])?;

    let expected = ArrayBuf::new(
        vec![
            0, 4, 8, //
            12, 16, 20, //
            1, 5, 9, //
            13, 17, 21, //
            //
            2, 6, 10, //
            14, 18, 22, //
            3, 7, 11, //
            15, 19, 23, //
        ],
        shape![4, 2, 3],
    )?;

    let actual = input.transpose(Some(axes![2, 0, 1]))?;
    assert!(expected.eq(actual)?.all()?);

    Ok(())
}
