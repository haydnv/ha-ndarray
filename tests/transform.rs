use ha_ndarray::*;
use std::iter;

#[test]
fn test_broadcast_small() -> Result<(), Error> {
    let data = vec![5, 1];
    let strides = ArrayBuf::new(data.to_vec(), shape![2])?.broadcast(shape![1, 2])?;
    assert_eq!(&*strides.buffer()?.to_slice()?, &data);
    Ok(())
}

#[test]
fn test_broadcast_large() -> Result<(), Error> {
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
        expected.buffer()?.to_slice()?,
        actual.buffer()?.to_slice()?
    );

    Ok(())
}

#[test]
fn test_slice_and_write() -> Result<(), Error> {
    let mut input = ArrayBuf::constant(0, shape![512])?;
    let mut slice = input.as_mut().slice(range![(128..256).into()])?;
    slice.write_value(1)?;

    let expected = iter::repeat(0)
        .take(128)
        .chain(iter::repeat(1).take(128))
        .chain(iter::repeat(0).take(256))
        .collect::<Vec<_>>();

    let actual = input.buffer()?.to_slice()?;

    assert_eq!(expected, &*actual);

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

#[test]
fn test_offsets_to_coords() -> Result<(), Error> {
    let coords = ArrayBuf::new(stackvec![0, 1], shape![1, 2])?;
    let strides = ArrayBuf::new(vec![5, 1], shape![2])?.broadcast(coords.shape().into())?;
    let offsets = coords.mul(strides).map(ArrayAccess::from)?;
    let offsets = offsets.sum(axes![1], false)?;
    assert_eq!(offsets.buffer()?.to_slice()?.into_vec(), vec![1]);
    Ok(())
}
