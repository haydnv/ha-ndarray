#[cfg(test)]
mod tests {
    use custos_math::Matrix;

    #[test]
    fn test_cpu() {
        use custos::CPU;

        let device = CPU::new();

        let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.]));
        let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.]));

        let c = a.gemm(&b);

        assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn test_opencl() {
        use custos::CLDevice;

        let device = CLDevice::new(0).expect("OpenCL device");

        let a = Matrix::from((&device, (2, 3), [1., 2., 3., 4., 5., 6.]));
        let b = Matrix::from((&device, (3, 2), [6., 5., 4., 3., 2., 1.]));

        let c = a.gemm(&b);

        assert_eq!(c.read(), vec![20., 14., 56., 41.,]);
    }
}
