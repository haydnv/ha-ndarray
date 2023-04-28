use ha_ndarray::{ArrayBase, Error, NDArray, NDArrayCompareScalar, NDArrayReduce};

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    for x in 1..9 {
        let data = vec![1; 10_usize.pow(x)];
        let array = ArrayBase::<i32>::new(vec![data.len()], data)?;
        assert_eq!(array.size() as i32, array.sum()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_axis() -> Result<(), Error> {
    let shapes = vec![
        vec![5],
        vec![2, 3, 4],
        vec![7],
        vec![2, 3, 129],
        vec![299],
        vec![51, 1, 13, 7, 64, 10, 2],
    ];

    for shape in shapes {
        let size = shape.iter().product();
        let array = ArrayBase::<u32>::new(shape.to_vec(), vec![1; size])?;
        for x in 0..shape.len() {
            let sum = array.sum_axis(x)?;
            let eq = sum.eq_scalar(shape[x] as u32)?;
            assert!(eq.all()?);
        }
    }

    Ok(())
}
