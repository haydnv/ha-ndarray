use ha_ndarray::{ArrayBase, Error, NDArray, NDArrayReduce};

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    for x in 1..9 {
        let data = vec![1; 10_usize.pow(x)];
        let array = ArrayBase::<i32>::from_vec(vec![data.len()], data)?;
        assert_eq!(array.size() as i32, array.sum()?);
    }

    Ok(())
}
