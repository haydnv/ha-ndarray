use ha_ndarray::{ArrayBase, Error, NDArray, NDArrayCompare, NDArrayReduce, NDArrayTransform};

#[test]
fn test_slice_1d() -> Result<(), Error> {
    let input = ArrayBase::from_vec(vec![4], (0..4).into_iter().collect())?;
    let expected = ArrayBase::from_vec(vec![2], (1..3).into_iter().collect())?;
    let actual = input.slice(vec![(1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(&actual)?.all()?);

    Ok(())
}

#[test]
fn test_slice_2d() -> Result<(), Error> {
    let input = ArrayBase::from_vec(vec![4, 3], (0..12).into_iter().collect())?;

    let expected = ArrayBase::from_vec(
        vec![2, 3],
        [
            3, 4, 5, //
            6, 7, 8, //
        ]
        .into(),
    )?;

    let actual = input.slice(vec![(1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(&actual)?.all()?);

    Ok(())
}

#[test]
fn test_slice_3d() -> Result<(), Error> {
    let input = ArrayBase::from_vec(vec![4, 3, 2], (0..24).into_iter().collect())?;

    let expected = ArrayBase::from_vec(vec![2, 2], [8, 9, 10, 11].into())?;

    let actual = input.slice(vec![1.into(), (1..3).into()])?;

    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(&actual)?.all()?);

    Ok(())
}

#[test]
fn test_transpose_2d() -> Result<(), Error> {
    let input = ArrayBase::from_vec(vec![2, 3], (0..6).into_iter().collect())?;

    let expected = ArrayBase::from_vec(
        vec![3, 2],
        vec![
            0, 3, //
            1, 4, //
            2, 5, //
        ],
    )?;

    let actual = input.transpose(None)?;
    assert_eq!(expected.shape(), actual.shape());
    assert!(expected.eq(&actual)?.all()?);

    Ok(())
}

#[test]
fn test_transpose_3d() -> Result<(), Error> {
    let input = ArrayBase::from_vec(vec![2, 3, 4], (0..24).into_iter().collect())?;

    let expected = ArrayBase::from_vec(
        vec![4, 2, 3],
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
    )?;

    let actual = input.transpose(Some(vec![2, 0, 1]))?;
    assert!(expected.eq(&actual)?.all()?);

    Ok(())
}
