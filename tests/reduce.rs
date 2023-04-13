use ha_ndarray::{ArrayBase, Error, NDArray, NDArrayCompareScalar, NDArrayReduce};

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    for x in 1..9 {
        let data = vec![1; 10_usize.pow(x)];
        let array = ArrayBase::<i32>::from_vec(vec![data.len()], data)?;
        assert_eq!(array.size() as i32, array.sum()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_axis() -> Result<(), Error> {
    let shape = vec![2, 3, 4];
    let array = ArrayBase::<u32>::constant(shape.to_vec(), 1);
    for x in 0..3 {
        let sum = array.sum_axis(x)?;
        let eq = sum.eq(shape[x] as u32)?;
        assert!(eq.all()?);
    }

    Ok(())
}
