use ha_ndarray::{
    ArrayBase, Error, NDArray, NDArrayCompare, NDArrayRead, NDArrayReduce, NDArrayTransform,
};

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

    let actual = input.transpose(None)?.copy()?;
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

    let actual = input.transpose(Some(vec![2, 0, 1]))?.copy()?;
    assert!(expected.eq(&actual)?.all()?);

    Ok(())
}
