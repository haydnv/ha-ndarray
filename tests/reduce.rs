use ha_ndarray::*;

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    for x in 1..9 {
        let array = ArrayBuf::constant(1, shape![10_usize.pow(x)]).map(ArrayAccess::from)?;
        assert_eq!(array.size() as i32, array.sum_all()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_range_axis() -> Result<(), Error> {
    let array = ArrayOp::range(0, 10, shape![1, 2, 5]).map(ArrayAccess::from)?;
    let actual = array.sum(axes![1], false)?;
    let expected = ArrayBuf::new(vec![5, 7, 9, 11, 13], shape![1, 5])?;

    println!("actual: {:?}", actual.buffer()?.to_slice()?);

    assert!(actual.eq(expected)?.all()?);
    Ok(())
}

#[test]
fn test_reduce_sum_axis() -> Result<(), Error> {
    let shapes = vec![
        shape![5],
        shape![2, 3, 4],
        shape![7],
        shape![2, 3, 129],
        shape![299],
        shape![51, 1, 13, 7, 64, 10, 2],
    ];

    for shape in shapes {
        let array = ArrayBuf::constant(1u32, shape.clone()).map(ArrayAccess::from)?;

        for x in 0..shape.len() {
            let expected = shape[x] as u32;
            let actual = array.clone().sum(axes![x], false)?;
            let eq = actual.eq_scalar(expected)?;
            assert!(eq.all()?);
        }
    }

    Ok(())
}
