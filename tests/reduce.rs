use ha_ndarray::{ArrayBase, Error, NDArrayReduce};

#[test]
fn test_reduce_sum_all_size_log2() -> Result<(), Error> {
    for x in 1..8 {
        let data = (0..2_i32.pow(x)).into_iter().collect::<Vec<i32>>();
        let expected: i32 = data.iter().sum();

        let array = ArrayBase::<i32>::from_vec(vec![data.len()], data)?;
        assert_eq!(expected, array.sum()?);
    }

    Ok(())
}
