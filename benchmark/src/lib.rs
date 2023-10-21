extern crate ocl;
use ha_ndarray::*;
use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use rand::Rng;
use std::time::Instant;

#[pymodule]
#[pyo3(name = "ha_ndarray")]
fn rustlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ndarray_test, m)?)?;
    m.add_function(wrap_pyfunction!(ha_ndarray_test, m)?)?;
    Ok(())
}

const SMALL_NUMBER: f64 = 1e-10;

enum DynamicArray {
    U8(Array2<u8>),
    U16(Array2<u16>),
    U32(Array2<u32>),
    U64(Array2<u64>),
    F32(Array2<f32>),
    F64(Array2<f64>),
}

fn random_array(size: usize, datatype: &str) -> DynamicArray {
    match datatype {
        "uint8" => DynamicArray::U8(Array2::random((size, size), Uniform::new(0, u8::MAX))),
        "uint16" => DynamicArray::U16(Array2::random((size, size), Uniform::new(0, u16::MAX))),
        "uint32" => DynamicArray::U32(Array2::random((size, size), Uniform::new(0, u32::MAX))),
        "uint64" => DynamicArray::U64(Array2::random((size, size), Uniform::new(0, u64::MAX))),
        "float32" => DynamicArray::F32(Array2::random((size, size), Uniform::new(0., 1.))),
        "float64" => DynamicArray::F64(Array2::random((size, size), Uniform::new(0., 1.))),
        _ => DynamicArray::F32(Array2::random((size, size), Uniform::new(0., 1.))),
    }
}

fn replace_zeros(arr: &mut DynamicArray) {
    match arr {
        DynamicArray::U8(arr_data) => {
            for element in arr_data.iter_mut() {
                if *element == 0 {
                    *element = 1; // replace with 1 for unsigned integers
                }
            }
        }
        DynamicArray::U16(arr_data) => {
            for element in arr_data.iter_mut() {
                if *element == 0 {
                    *element = 1;
                }
            }
        }
        DynamicArray::U32(arr_data) => {
            for element in arr_data.iter_mut() {
                if *element == 0 {
                    *element = 1;
                }
            }
        }
        DynamicArray::U64(arr_data) => {
            for element in arr_data.iter_mut() {
                if *element == 0 {
                    *element = 1;
                }
            }
        }
        DynamicArray::F32(arr_data) => {
            for element in arr_data.iter_mut() {
                if *element == 0.0 {
                    *element = SMALL_NUMBER as f32; // replace with small number for floating-point numbers
                }
            }
        }
        DynamicArray::F64(arr_data) => {
            for element in arr_data.iter_mut() {
                if *element == 0.0 {
                    *element = SMALL_NUMBER;
                }
            }
        }
    }
}

fn add_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(arr_a), DynamicArray::U8(arr_b)) => Ok(DynamicArray::U8(arr_a + arr_b)),
        (DynamicArray::U16(arr_a), DynamicArray::U16(arr_b)) => {
            Ok(DynamicArray::U16(arr_a + arr_b))
        }
        (DynamicArray::U32(arr_a), DynamicArray::U32(arr_b)) => {
            Ok(DynamicArray::U32(arr_a + arr_b))
        }
        (DynamicArray::U64(arr_a), DynamicArray::U64(arr_b)) => {
            Ok(DynamicArray::U64(arr_a + arr_b))
        }
        (DynamicArray::F32(arr_a), DynamicArray::F32(arr_b)) => {
            Ok(DynamicArray::F32(arr_a + arr_b))
        }
        (DynamicArray::F64(arr_a), DynamicArray::F64(arr_b)) => {
            Ok(DynamicArray::F64(arr_a + arr_b))
        }
        _ => Err("nah dawg"),
    }
}

fn sub_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(arr_a), DynamicArray::U8(arr_b)) => Ok(DynamicArray::U8(arr_a - arr_b)),
        (DynamicArray::U16(arr_a), DynamicArray::U16(arr_b)) => {
            Ok(DynamicArray::U16(arr_a - arr_b))
        }
        (DynamicArray::U32(arr_a), DynamicArray::U32(arr_b)) => {
            Ok(DynamicArray::U32(arr_a - arr_b))
        }
        (DynamicArray::U64(arr_a), DynamicArray::U64(arr_b)) => {
            Ok(DynamicArray::U64(arr_a - arr_b))
        }
        (DynamicArray::F32(arr_a), DynamicArray::F32(arr_b)) => {
            Ok(DynamicArray::F32(arr_a - arr_b))
        }
        (DynamicArray::F64(arr_a), DynamicArray::F64(arr_b)) => {
            Ok(DynamicArray::F64(arr_a - arr_b))
        }
        _ => Err("nah dawg"),
    }
}

fn mul_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(arr_a), DynamicArray::U8(arr_b)) => Ok(DynamicArray::U8(arr_a * arr_b)),
        (DynamicArray::U16(arr_a), DynamicArray::U16(arr_b)) => {
            Ok(DynamicArray::U16(arr_a * arr_b))
        }
        (DynamicArray::U32(arr_a), DynamicArray::U32(arr_b)) => {
            Ok(DynamicArray::U32(arr_a * arr_b))
        }
        (DynamicArray::U64(arr_a), DynamicArray::U64(arr_b)) => {
            Ok(DynamicArray::U64(arr_a * arr_b))
        }
        (DynamicArray::F32(arr_a), DynamicArray::F32(arr_b)) => {
            Ok(DynamicArray::F32(arr_a * arr_b))
        }
        (DynamicArray::F64(arr_a), DynamicArray::F64(arr_b)) => {
            Ok(DynamicArray::F64(arr_a * arr_b))
        }
        _ => Err("nah dawg"),
    }
}

fn div_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(arr_a), DynamicArray::U8(arr_b)) => Ok(DynamicArray::U8(arr_a / arr_b)),
        (DynamicArray::U16(arr_a), DynamicArray::U16(arr_b)) => {
            Ok(DynamicArray::U16(arr_a / arr_b))
        }
        (DynamicArray::U32(arr_a), DynamicArray::U32(arr_b)) => {
            Ok(DynamicArray::U32(arr_a / arr_b))
        }
        (DynamicArray::U64(arr_a), DynamicArray::U64(arr_b)) => {
            Ok(DynamicArray::U64(arr_a / arr_b))
        }
        (DynamicArray::F32(arr_a), DynamicArray::F32(arr_b)) => {
            Ok(DynamicArray::F32(arr_a / arr_b))
        }
        (DynamicArray::F64(arr_a), DynamicArray::F64(arr_b)) => {
            Ok(DynamicArray::F64(arr_a / arr_b))
        }
        _ => Err("nah dawg"),
    }
}

fn dot_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(arr_a), DynamicArray::U8(arr_b)) => {
            Ok(DynamicArray::U8(arr_a.dot(arr_b)))
        }
        (DynamicArray::U16(arr_a), DynamicArray::U16(arr_b)) => {
            Ok(DynamicArray::U16(arr_a.dot(arr_b)))
        }
        (DynamicArray::U32(arr_a), DynamicArray::U32(arr_b)) => {
            Ok(DynamicArray::U32(arr_a.dot(arr_b)))
        }
        (DynamicArray::U64(arr_a), DynamicArray::U64(arr_b)) => {
            Ok(DynamicArray::U64(arr_a.dot(arr_b)))
        }
        (DynamicArray::F32(arr_a), DynamicArray::F32(arr_b)) => {
            Ok(DynamicArray::F32(arr_a.dot(arr_b)))
        }
        (DynamicArray::F64(arr_a), DynamicArray::F64(arr_b)) => {
            Ok(DynamicArray::F64(arr_a.dot(arr_b)))
        }
        _ => Err("nah dawg"),
    }
}

#[pyfunction]
pub fn ndarray_test(py: Python) -> PyResult<PyObject> {
    let labels: Vec<&str> = vec!["uint8", "uint16", "uint32", "uint64", "float32", "float64"];
    let sizes: Vec<usize> = (1..=11).step_by(2).map(|x| 2_usize.pow(x)).collect();
    let operations: Vec<&str> = vec!["add", "sub", "mul", "div", "dot"];

    let timings_ndarrays = PyDict::new(py);

    for op in &operations {
        let mut timing_data = Vec::new();
        for label in &labels {
            let mut label_data = Vec::new();
            for &size in &sizes {
                println!(
                    "Rust ndarray - type: {}, size: {}, operation {}",
                    label, size, op
                );
                let mut results = Vec::new();
                for _ in 0..((sizes.last().unwrap() / &size) + 2) {
                    let mut a = random_array(size, label);
                    let mut b = random_array(size, label);

                    let start: Instant;

                    match op.as_ref() {
                        "add" => {
                            start = std::time::Instant::now();
                            let _ = add_arrays(&a, &b);
                        }
                        "sub" => {
                            start = std::time::Instant::now();
                            let _ = sub_arrays(&a, &b);
                        }
                        "mul" => {
                            start = std::time::Instant::now();
                            let _ = mul_arrays(&a, &b);
                        }
                        "div" => {
                            replace_zeros(&mut a);
                            replace_zeros(&mut b);
                            start = std::time::Instant::now();
                            let _ = div_arrays(&a, &b);
                        }
                        "dot" => {
                            start = std::time::Instant::now();
                            let _ = dot_arrays(&a, &b);
                        }
                        _ => {
                            start = std::time::Instant::now();
                        }
                    }

                    let duration = start.elapsed();
                    results.push(duration.as_secs_f64());
                }
                let average_time: f64 = results.iter().sum::<f64>() / results.len() as f64;
                label_data.push(average_time);
            }
            timing_data.push(label_data.into_pyarray(py).to_owned());
        }
        timings_ndarrays.set_item(op, timing_data)?;
    }

    Ok(timings_ndarrays.to_object(py))
}

enum DynamicArrayHa {
    U8(ArrayBase<Vec<u8>>),
    U16(ArrayBase<Vec<u16>>),
    U32(ArrayBase<Vec<u32>>),
    U64(ArrayBase<Vec<u64>>),
    F32(ArrayBase<Vec<f32>>),
    F64(ArrayBase<Vec<f64>>),
}

fn random_array_ha(size: usize, datatype: &str, context: &Context) -> DynamicArrayHa {
    let mut rng = rand::thread_rng();
    match datatype {
        "uint8" => {
            // nonzeros for the div function
            let data: Vec<u8> = (0..size * size)
                .map(|_| loop {
                    let val = rng.gen::<u8>();
                    if val != 0 {
                        break val;
                    }
                })
                .collect();
            DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "uint16" => {
            let data: Vec<u16> = (0..size * size)
                .map(|_| loop {
                    let val = rng.gen::<u16>();
                    if val != 0 {
                        break val;
                    }
                })
                .collect();
            DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "uint32" => {
            let data: Vec<u32> = (0..size * size)
                .map(|_| loop {
                    let val = rng.gen::<u32>();
                    if val != 0 {
                        break val;
                    }
                })
                .collect();
            DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "uint64" => {
            let data: Vec<u64> = (0..size * size)
                .map(|_| loop {
                    let val = rng.gen::<u64>();
                    if val != 0 {
                        break val;
                    }
                })
                .collect();
            DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "float32" => {
            let data: Vec<f32> = (0..size * size)
                .map(|_| loop {
                    let val = rng.gen::<f32>();
                    if val != 0.0 {
                        break val;
                    }
                })
                .collect();
            DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "float64" => {
            let data: Vec<f64> = (0..size * size)
                .map(|_| loop {
                    let val = rng.gen::<f64>();
                    if val != 0.0 {
                        break val;
                    }
                })
                .collect();
            DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        _ => panic!("Invalid datatype"),
    }
}

fn add_arrays_ha(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(arr_a), DynamicArrayHa::U8(arr_b)) => {
            let result_op = arr_a.add(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U16(arr_a), DynamicArrayHa::U16(arr_b)) => {
            let result_op = arr_a.add(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U32(arr_a), DynamicArrayHa::U32(arr_b)) => {
            let result_op = arr_a.add(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U64(arr_a), DynamicArrayHa::U64(arr_b)) => {
            let result_op = arr_a.add(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F32(arr_a), DynamicArrayHa::F32(arr_b)) => {
            let result_op = arr_a.add(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F64(arr_a), DynamicArrayHa::F64(arr_b)) => {
            let result_op = arr_a.add(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        _ => Err("nah dawg"),
    }
}

fn sub_arrays_ha(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(arr_a), DynamicArrayHa::U8(arr_b)) => {
            let result_op = arr_a.sub(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U16(arr_a), DynamicArrayHa::U16(arr_b)) => {
            let result_op = arr_a.sub(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U32(arr_a), DynamicArrayHa::U32(arr_b)) => {
            let result_op = arr_a.sub(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U64(arr_a), DynamicArrayHa::U64(arr_b)) => {
            let result_op = arr_a.sub(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F32(arr_a), DynamicArrayHa::F32(arr_b)) => {
            let result_op = arr_a.sub(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F64(arr_a), DynamicArrayHa::F64(arr_b)) => {
            let result_op = arr_a.sub(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        _ => Err("nah dawg"),
    }
}

fn mul_arrays_ha(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(arr_a), DynamicArrayHa::U8(arr_b)) => {
            let result_op = arr_a.mul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U16(arr_a), DynamicArrayHa::U16(arr_b)) => {
            let result_op = arr_a.mul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U32(arr_a), DynamicArrayHa::U32(arr_b)) => {
            let result_op = arr_a.mul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U64(arr_a), DynamicArrayHa::U64(arr_b)) => {
            let result_op = arr_a.mul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F32(arr_a), DynamicArrayHa::F32(arr_b)) => {
            let result_op = arr_a.mul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F64(arr_a), DynamicArrayHa::F64(arr_b)) => {
            let result_op = arr_a.mul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        _ => Err("nah dawg"),
    }
}

fn div_arrays_ha(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(arr_a), DynamicArrayHa::U8(arr_b)) => {
            let result_op = arr_a.div(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U16(arr_a), DynamicArrayHa::U16(arr_b)) => {
            let result_op = arr_a.div(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U32(arr_a), DynamicArrayHa::U32(arr_b)) => {
            let result_op = arr_a.div(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U64(arr_a), DynamicArrayHa::U64(arr_b)) => {
            let result_op = arr_a.div(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F32(arr_a), DynamicArrayHa::F32(arr_b)) => {
            let result_op = arr_a.div(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F64(arr_a), DynamicArrayHa::F64(arr_b)) => {
            let result_op = arr_a.div(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        _ => Err("nah dawg"),
    }
}

fn dot_arrays_ha(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(arr_a), DynamicArrayHa::U8(arr_b)) => {
            let result_op = arr_a.matmul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U16(arr_a), DynamicArrayHa::U16(arr_b)) => {
            let result_op = arr_a.matmul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U32(arr_a), DynamicArrayHa::U32(arr_b)) => {
            let result_op = arr_a.matmul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::U64(arr_a), DynamicArrayHa::U64(arr_b)) => {
            let result_op = arr_a.matmul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F32(arr_a), DynamicArrayHa::F32(arr_b)) => {
            let result_op = arr_a.matmul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        (DynamicArrayHa::F64(arr_a), DynamicArrayHa::F64(arr_b)) => {
            let result_op = arr_a.matmul(arr_b).unwrap();
            let queue = Queue::new(context.clone(), size * size).unwrap();
            let result_array = if let BufferConverter::Host(SliceConverter::Vec(data)) =
                result_op.read(&queue).unwrap()
            {
                data
            } else {
                panic!("Unexpected buffer type");
            };
            Ok(DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], result_array)
                    .unwrap(),
            ))
        }
        _ => Err("nah dawg"),
    }
}

fn ha_error_to_py_err(err: ha_ndarray::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("ha_ndarray error: {}", err))
}

#[pyfunction]
pub fn ha_ndarray_test(py: Python) -> PyResult<PyObject> {
    let labels: Vec<&str> = vec!["uint8", "uint16", "uint32", "uint64", "float32", "float64"];
    let sizes: Vec<usize> = (1..=11).step_by(2).map(|x| 2_usize.pow(x)).collect();
    let operations: Vec<&str> = vec!["add", "sub", "mul", "div", "dot"];
    let context = Context::default().map_err(ha_error_to_py_err)?;

    let timings_ha_ndarrays = PyDict::new(py);

    for op in &operations {
        let mut timing_data = Vec::new();
        for label in &labels {
            let mut label_data = Vec::new();
            for &size in &sizes {
                println!(
                    "Rust ha-ndarray - type: {}, size: {}, operation {}",
                    label, size, op
                );
                let mut results = Vec::new();
                for _ in 0..((sizes.last().unwrap() / &size) + 2) {
                    let a = random_array_ha(size, label, &context);
                    let b = random_array_ha(size, label, &context);

                    let start: Instant;
                    match op.as_ref() {
                        "add" => {
                            start = std::time::Instant::now();
                            let _ = add_arrays_ha(a, b, size, &context);
                        }
                        "sub" => {
                            start = std::time::Instant::now();
                            let _ = sub_arrays_ha(a, b, size, &context);
                        }
                        "mul" => {
                            start = std::time::Instant::now();
                            let _ = mul_arrays_ha(a, b, size, &context);
                        }
                        "div" => {
                            start = std::time::Instant::now();
                            let _ = div_arrays_ha(a, b, size, &context);
                        }
                        "dot" => {
                            start = std::time::Instant::now();
                            let _ = dot_arrays_ha(a, b, size, &context);
                        }
                        _ => {
                            start = std::time::Instant::now();
                        }
                    }

                    let duration = start.elapsed();
                    results.push(duration.as_secs_f64());
                }
                let average_time: f64 = results.iter().sum::<f64>() / results.len() as f64;
                label_data.push(average_time);
            }
            timing_data.push(label_data.into_pyarray(py).to_owned());
        }
        timings_ha_ndarrays.set_item(op, timing_data)?;
    }

    Ok(timings_ha_ndarrays.to_object(py))
}
