use ha_ndarray::*;

#[test]
fn test_matmul() -> Result<(), Error> {
    let shapes = [
        (vec![2, 3], vec![3, 4], vec![2, 4]),
        (vec![2, 2, 3], vec![2, 3, 4], vec![2, 2, 4]),
        (vec![9, 7], vec![7, 12], vec![9, 12]),
        (vec![16, 8], vec![8, 24], vec![16, 24]),
        (vec![2, 9], vec![9, 1], vec![2, 1]),
        (vec![15, 26], vec![26, 37], vec![15, 37]),
        (vec![3, 15, 26], vec![3, 26, 37], vec![3, 15, 37]),
        (vec![8, 44, 1], vec![8, 1, 98], vec![8, 44, 98]),
    ];

    for (left_shape, right_shape, output_shape) in shapes {
        let context = Context::new(0, 0, None)?;

        let left = vec![1.; left_shape.iter().product()];
        let left = ArrayBase::with_context(context.clone(), left_shape, left)?;
        let right = vec![1.; right_shape.iter().product()];
        let right = ArrayBase::with_context(context.clone(), right_shape, right)?;

        let expected = *left.shape().last().unwrap();
        let actual = left.matmul(right)?;
        assert_eq!(actual.shape(), output_shape);

        let eq = actual.eq_scalar(expected as f32)?;
        assert!(eq.all()?);
    }

    Ok(())
}
