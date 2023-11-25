use ha_ndarray::*;

#[test]
fn test_matmul_small() -> Result<(), Error> {
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
fn test_matmul_large() -> Result<(), Error> {
    let shapes: [(Shape, Shape, Shape); 8] = [
        (shape![2, 3], shape![3, 4], shape![2, 4]),
        (shape![2, 2, 3], shape![2, 3, 4], shape![2, 2, 4]),
        (shape![9, 7], shape![7, 12], shape![9, 12]),
        (shape![16, 8], shape![8, 24], shape![16, 24]),
        (shape![2, 9], shape![9, 1], shape![2, 1]),
        (shape![15, 26], shape![26, 37], shape![15, 37]),
        (shape![3, 15, 26], shape![3, 26, 37], shape![3, 15, 37]),
        (shape![8, 44, 1], shape![8, 1, 98], shape![8, 44, 98]),
    ];

    for (left_shape, right_shape, output_shape) in shapes {
        println!("{left_shape:?} @ {right_shape:?}");

        let left = vec![1.; left_shape.iter().product()];
        let left = ArrayBuf::new(left, left_shape)?;
        let right = vec![1.; right_shape.iter().product()];
        let right = ArrayBuf::new(right, right_shape)?;

        let expected = *left.shape().last().unwrap();
        let actual = left.matmul(right)?;
        assert_eq!(actual.shape(), output_shape.as_slice());

        let eq = actual.eq_scalar(expected as f32)?;
        assert!(eq.all()?);
    }

    Ok(())
}
