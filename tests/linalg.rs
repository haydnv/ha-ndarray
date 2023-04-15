use ha_ndarray::{
    ArrayBase, Error, MatrixMath, NDArray, NDArrayCompareScalar, NDArrayRead, NDArrayReduce,
};

#[test]
fn test_matmul() -> Result<(), Error> {
    let shapes = [
        (vec![2, 3], vec![3, 4], vec![2, 4]),
        (vec![2, 2, 3], vec![2, 3, 4], vec![2, 2, 4]),
    ];

    for (left_shape, right_shape, output_shape) in shapes {
        let left = ArrayBase::constant(left_shape, 1.);
        let right = ArrayBase::constant(right_shape, 1.);
        let actual = left.matmul(&right)?.copy()?;
        assert_eq!(actual.shape(), output_shape);
        println!("{:?}", actual.to_vec());
        assert!(actual.eq(left.shape()[left.ndim() - 1] as f32)?.all()?);
    }

    Ok(())
}
