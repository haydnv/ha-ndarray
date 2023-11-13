use ha_ndarray::*;

#[test]
fn test_broadcast() -> Result<(), Error> {
    let input = Array::new(vec![1; 600].into(), shape![300, 1, 2])?;
    let output = input.broadcast(shape![300, 250, 2])?;
    assert_eq!(output.shape(), &[300, 250, 2]);
    assert_eq!(output.sum_all()?, 150000);
    Ok(())
}
