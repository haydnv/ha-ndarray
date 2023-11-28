use ha_ndarray::*;

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
