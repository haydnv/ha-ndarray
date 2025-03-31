use ha_ndarray::{
    shape, Access, AccessBuf, Array, ArrayBuf, Buffer, Error, Float, NDArrayMath,
    NDArrayMathScalar, NDArrayUnary,
};

// the accuracy of these operations is covered by the other test modules
// the purpose of this module is to test that `Array`s with lifetime bounds compile successfully

fn logit<A, T>(p: Array<T, A>) -> Result<Array<T, impl Access<T>>, Error>
where
    A: Access<T> + Clone,
    T: Float + std::ops::Neg<Output = T>,
{
    p.clone().div(p.add_scalar(-T::ONE)?)?.ln()
}

#[test]
fn test_as_ref() -> Result<(), Error> {
    let a = ArrayBuf::<f32, Buffer<f32>>::new(vec![1.].into(), shape![1])?;
    let a = a.as_ref::<AccessBuf<&Buffer<f32>>>();
    logit(a)?;
    Ok(())
}
