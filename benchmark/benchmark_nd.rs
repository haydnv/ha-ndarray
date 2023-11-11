use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

const SMALL_NUMBER: f64 = 1e-10;

#[derive(Debug)]
enum DynamicArray {
    U8(Array2<u8>),
    U16(Array2<u16>),
    U32(Array2<u32>),
    U64(Array2<u64>),
    F32(Array2<f32>),
    F64(Array2<f64>),
}

fn max_value(size: usize, bytes: u32) -> u64 {
    let max_val_1: u64 = (2_u64.pow(bytes)) / size as u64;
    let max_val_2: u64 = bytes as u64 / 2;
    let max_val: u64 = max_val_1.min(max_val_2);
    if max_val == 0 {
        // prevents under/overflow, but maybe this should thin out
        // the array instead; cos does the compiler recognise
        // zeors in everything here?
        1
    } else {
        max_val - 1
    }
}

fn random_array(size: usize, datatype: &str) -> DynamicArray {
    match datatype {
        "uint8" => {
            let max_val = max_value(size, 8);
            DynamicArray::U8(Array2::random((size, size), Uniform::new(0, max_val as u8)))
        }
        "uint16" => {
            let max_val = max_value(size, 16);
            DynamicArray::U16(Array2::random(
                (size, size),
                Uniform::new(0, max_val as u16),
            ))
        }
        "uint32" => {
            let max_val = max_value(size, 32);
            DynamicArray::U32(Array2::random(
                (size, size),
                Uniform::new(0, max_val as u32),
            ))
        }
        "uint64" => {
            let max_val = max_value(size, 32);
            DynamicArray::U64(Array2::random(
                (size, size),
                Uniform::new(0, max_val as u64),
            ))
        }
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

pub fn ndarray_test(
    data_types: Vec<&'static str>,
    operations: Vec<&'static str>,
    sizes: Vec<usize>,
) -> HashMap<&'static str, Vec<Vec<f64>>> {
    let mut timings = HashMap::new();

    for op in operations {
        let mut timing_data = Vec::new();
        for dtype in &data_types {
            let mut label_data = Vec::new();
            for &size in &sizes {
                let mut results = Vec::new();
                for _ in 0..((sizes.last().unwrap() / size) + 2) {
                    let mut a = random_array(size, dtype);
                    let mut b = random_array(size, dtype);

                    let start;

                    let _ = match op {
                        "add" => {
                            start = Instant::now();
                            add_arrays(&a, &b)
                        }
                        "sub" => {
                            let c = add_arrays(&a, &b).unwrap();
                            start = Instant::now();
                            sub_arrays(&c, &a)
                        }
                        "mul" => {
                            start = Instant::now();
                            mul_arrays(&a, &b)
                        }

                        "div" => {
                            replace_zeros(&mut a);
                            replace_zeros(&mut b);
                            start = Instant::now();
                            div_arrays(&a, &b)
                        }
                        "dot" => {
                            start = Instant::now();
                            dot_arrays(&a, &b)
                        }
                        _ => panic!("Unsupported operation: {}", op),
                    };

                    results.push(start.elapsed().as_secs_f64());
                }
                let average_time: f64 = results.iter().sum::<f64>() / results.len() as f64;
                label_data.push(average_time.clone());

                // println!(
                //     "nd-array - type: {}, size: {}, operation: {}, time: {}",
                //     dtype, size, op, average_time
                // );
                print!(".");
                io::stdout().flush().unwrap();
            }
            timing_data.push(label_data);
        }
        timings.insert(op, timing_data);
    }
    println!(" ");
    timings
}
