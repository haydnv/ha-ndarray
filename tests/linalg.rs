use ha_ndarray::*;

#[test]
fn test_diag() -> Result<(), Error> {
    let x = ArrayOp::range(0, 9, shape![3, 3])?;
    let diag = x.diag()?;
    assert_eq!(&*diag.buffer()?.to_slice()?, &[0, 4, 8]);
    Ok(())
}

#[test]
fn test_matmul_12x20() -> Result<(), Error> {
    let l = ArrayBuf::new((0..12).into_iter().collect::<Vec<_>>(), shape![3, 4])?;
    let r = ArrayBuf::new((0..20).into_iter().collect::<Vec<_>>(), shape![4, 5])?;

    let actual = l.matmul(r)?;

    let expected = ArrayBuf::new(
        vec![
            70, 76, 82, 88, 94, 190, 212, 234, 256, 278, 310, 348, 386, 424, 462,
        ],
        shape![3, 5],
    )?;

    assert_eq!(actual.shape(), expected.shape());

    let eq = actual.eq(expected)?;
    assert!(eq.all()?);

    Ok(())
}

#[test]
fn test_dot_product() -> Result<(), Error> {
    let buffer = (0..16).into_iter().collect::<Vec<_>>();
    let vectors = ArrayBuf::<i32, Buffer<i32>>::new(buffer.to_vec().into(), shape![4, 1, 4])?;

    let l: ArrayBuf<i32, &Buffer<i32>> = vectors.as_ref();
    let r = l.clone().mt()?;

    let actual = l.matmul(r)?.buffer()?.to_slice()?.to_vec();

    let mut expected = Vec::with_capacity(4 * 4);
    for vector in buffer.chunks(4) {
        expected.push(vector.iter().zip(vector.iter()).map(|(l, r)| l * r).sum());
    }

    assert_eq!(actual, expected);

    Ok(())
}
