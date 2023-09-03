use std::sync::Arc;

use ha_ndarray::*;

#[test]
fn test_reduce_sum_all() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;

    for x in 1..9 {
        let data = vec![1; 10_usize.pow(x)];
        let host_array =
            ArrayBase::<Vec<i32>>::with_context(context.clone(), vec![data.len()], data)?;

        #[cfg(feature = "opencl")]
        {
            let cl_array = ArrayBase::<ocl::Buffer<i32>>::copy(&host_array)?;
            let op_array = cl_array + host_array.clone();
            assert_eq!((op_array.size() * 2) as i32, op_array.sum_all()?);
        }

        assert_eq!(host_array.size() as i32, host_array.sum_all()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_axis() -> Result<(), Error> {
    let context = Context::new(0, 0, None)?;

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
        let array = ArrayBase::<Arc<Vec<_>>>::with_context(
            context.clone(),
            shape.to_vec(),
            Arc::new(vec![1; size]),
        )?;

        for x in 0..shape.len() {
            let sum = array.clone().sum(vec![x], false)?;
            let eq = sum.eq_scalar(shape[x] as u32)?;
            assert!(eq.all()?);
        }
    }

    Ok(())
}
