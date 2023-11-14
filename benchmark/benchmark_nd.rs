use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Debug)]
enum DynamicArray {
    U8(Array2<u8>),
    U16(Array2<u16>),
    U32(Array2<u32>),
    U64(Array2<u64>),
    F32(Array2<f32>),
    F64(Array2<f64>),
}

fn random_array(size: usize, datatype: &str, op: &str) -> DynamicArray {
    let array: Array2<f32> = Array2::random((size, size), Uniform::new(0.0, 1.0));
    let mapped_array: Array2<f32>;
    // map array; 95% of number == 1, 5% of numbers are 2 (avoids over/underflows)
    let threshold: f32 = (((size * size) - size) / (size * size)) as f32;
    if op == "div" {
        mapped_array = array.mapv(|x| if x > threshold { 2.0 } else { 1.0 });
    } else {
        mapped_array = array.mapv(|x| if x > threshold { 0.0 } else { 1.0 });
    }

    match datatype {
        "uint8" => DynamicArray::U8(mapped_array.mapv(|x| x as u8)),
        "uint16" => DynamicArray::U16(mapped_array.mapv(|x| x as u16)),
        "uint32" => DynamicArray::U32(mapped_array.mapv(|x| x as u32)),
        "uint64" => DynamicArray::U64(mapped_array.mapv(|x| x as u64)),
        "float32" => DynamicArray::F32(mapped_array.mapv(|x| x as f32)),
        "float64" => DynamicArray::F64(mapped_array.mapv(|x| x as f64)),
        _ => DynamicArray::F32(mapped_array.mapv(|x| x as f32)),
    }
}

fn add_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(a), DynamicArray::U8(b)) => Ok(DynamicArray::U8(a + b)),
        (DynamicArray::U16(a), DynamicArray::U16(b)) => Ok(DynamicArray::U16(a + b)),
        (DynamicArray::U32(a), DynamicArray::U32(b)) => Ok(DynamicArray::U32(a + b)),
        (DynamicArray::U64(a), DynamicArray::U64(b)) => Ok(DynamicArray::U64(a + b)),
        (DynamicArray::F32(a), DynamicArray::F32(b)) => Ok(DynamicArray::F32(a + b)),
        (DynamicArray::F64(a), DynamicArray::F64(b)) => Ok(DynamicArray::F64(a + b)),
        _ => Err("nah dawg"),
    }
}

fn sub_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(a), DynamicArray::U8(b)) => Ok(DynamicArray::U8(a - b)),
        (DynamicArray::U16(a), DynamicArray::U16(b)) => Ok(DynamicArray::U16(a - b)),
        (DynamicArray::U32(a), DynamicArray::U32(b)) => Ok(DynamicArray::U32(a - b)),
        (DynamicArray::U64(a), DynamicArray::U64(b)) => Ok(DynamicArray::U64(a - b)),
        (DynamicArray::F32(a), DynamicArray::F32(b)) => Ok(DynamicArray::F32(a - b)),
        (DynamicArray::F64(a), DynamicArray::F64(b)) => Ok(DynamicArray::F64(a - b)),
        _ => Err("nah dawg"),
    }
}

fn mul_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(a), DynamicArray::U8(b)) => Ok(DynamicArray::U8(a * b)),
        (DynamicArray::U16(a), DynamicArray::U16(b)) => Ok(DynamicArray::U16(a * b)),
        (DynamicArray::U32(a), DynamicArray::U32(b)) => Ok(DynamicArray::U32(a * b)),
        (DynamicArray::U64(a), DynamicArray::U64(b)) => Ok(DynamicArray::U64(a * b)),
        (DynamicArray::F32(a), DynamicArray::F32(b)) => Ok(DynamicArray::F32(a * b)),
        (DynamicArray::F64(a), DynamicArray::F64(b)) => Ok(DynamicArray::F64(a * b)),
        _ => Err("nah dawg"),
    }
}

fn div_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(a), DynamicArray::U8(b)) => Ok(DynamicArray::U8(a / b)),
        (DynamicArray::U16(a), DynamicArray::U16(b)) => Ok(DynamicArray::U16(a / b)),
        (DynamicArray::U32(a), DynamicArray::U32(b)) => Ok(DynamicArray::U32(a / b)),
        (DynamicArray::U64(a), DynamicArray::U64(b)) => Ok(DynamicArray::U64(a / b)),
        (DynamicArray::F32(a), DynamicArray::F32(b)) => Ok(DynamicArray::F32(a / b)),
        (DynamicArray::F64(a), DynamicArray::F64(b)) => Ok(DynamicArray::F64(a / b)),
        _ => Err("nah dawg"),
    }
}

fn dot_arrays(a: &DynamicArray, b: &DynamicArray) -> Result<DynamicArray, &'static str> {
    match (a, b) {
        (DynamicArray::U8(a), DynamicArray::U8(b)) => Ok(DynamicArray::U8(a.dot(b))),
        (DynamicArray::U16(a), DynamicArray::U16(b)) => Ok(DynamicArray::U16(a.dot(b))),
        (DynamicArray::U32(a), DynamicArray::U32(b)) => Ok(DynamicArray::U32(a.dot(b))),
        (DynamicArray::U64(a), DynamicArray::U64(b)) => Ok(DynamicArray::U64(a.dot(b))),
        (DynamicArray::F32(a), DynamicArray::F32(b)) => Ok(DynamicArray::F32(a.dot(b))),
        (DynamicArray::F64(a), DynamicArray::F64(b)) => Ok(DynamicArray::F64(a.dot(b))),
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
                for _ in 0..10 {
                    //((sizes.last().unwrap() / size) + 2) {
                    let a = random_array(size, dtype, op);
                    let b = random_array(size, dtype, op);

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
                            // replace_zeros(&mut a);
                            // replace_zeros(&mut b);
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
