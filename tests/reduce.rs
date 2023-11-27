use ha_ndarray::*;

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    for x in 1..9 {
        let data = vec![1; 10_usize.pow(x)];
        let shape = shape![data.len()];
        let array = ArrayBuf::new(data, shape)?;
        assert_eq!(array.size() as i32, array.sum_all()?);
    }

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
        let array = ArrayBuf::constant(1u32, shape.clone())?;

        for x in 0..shape.len() {
            let sum = array.as_ref().sum(x, false)?;
            let eq = sum.eq_scalar(shape[x] as u32)?;
            assert!(eq.all()?);
        }
    }

    Ok(())
}
