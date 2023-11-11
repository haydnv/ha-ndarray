use ha_ndarray::*;
use rand::Rng;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

enum DynamicArrayHa {
    U8(ArrayBase<Vec<u8>>),
    U16(ArrayBase<Vec<u16>>),
    U32(ArrayBase<Vec<u32>>),
    U64(ArrayBase<Vec<u64>>),
    F32(ArrayBase<Vec<f32>>),
    F64(ArrayBase<Vec<f64>>),
}

impl Clone for DynamicArrayHa {
    fn clone(&self) -> Self {
        match self {
            DynamicArrayHa::U8(array) => DynamicArrayHa::U8(array.clone()),
            DynamicArrayHa::U16(array) => DynamicArrayHa::U16(array.clone()),
            DynamicArrayHa::U32(array) => DynamicArrayHa::U32(array.clone()),
            DynamicArrayHa::U64(array) => DynamicArrayHa::U64(array.clone()),
            DynamicArrayHa::F32(array) => DynamicArrayHa::F32(array.clone()),
            DynamicArrayHa::F64(array) => DynamicArrayHa::F64(array.clone()),
        }
    }
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

fn random_array(size: usize, datatype: &str, context: &Context) -> DynamicArrayHa {
    let mut rng = rand::thread_rng();
    let max_val = max_value(size, 8);
    match datatype {
        "uint8" => {
            // nonzeros for the div function
            let data: Vec<u8> = (0..size * size)
                .map(|_| rng.gen_range(1..=max_val as u8))
                .collect();
            DynamicArrayHa::U8(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "uint16" => {
            let data: Vec<u16> = (0..size * size)
                .map(|_| rng.gen_range(1..=max_val as u16))
                .collect();
            DynamicArrayHa::U16(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "uint32" => {
            let data: Vec<u32> = (0..size * size)
                .map(|_| rng.gen_range(1..=max_val as u32))
                .collect();
            DynamicArrayHa::U32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "uint64" => {
            let data: Vec<u64> = (0..size * size)
                .map(|_| rng.gen_range(1..=max_val as u64))
                .collect();
            DynamicArrayHa::U64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "float32" => {
            let data: Vec<f32> = (0..size * size).map(|_| rng.gen_range(0.0..=1.0)).collect();
            DynamicArrayHa::F32(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        "float64" => {
            let data: Vec<f64> = (0..size * size).map(|_| rng.gen_range(0.0..1.0)).collect();
            DynamicArrayHa::F64(
                ArrayBase::<Vec<_>>::with_context(context.clone(), vec![size, size], data).unwrap(),
            )
        }
        _ => panic!("Invalid datatype"),
    }
}

fn add_arrays(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(a), DynamicArrayHa::U8(b)) => {
            let result_op = a.add(b).unwrap();
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
        (DynamicArrayHa::U16(a), DynamicArrayHa::U16(b)) => {
            let result_op = a.add(b).unwrap();
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
        (DynamicArrayHa::U32(a), DynamicArrayHa::U32(b)) => {
            let result_op = a.add(b).unwrap();
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
        (DynamicArrayHa::U64(a), DynamicArrayHa::U64(b)) => {
            let result_op = a.add(b).unwrap();
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
        (DynamicArrayHa::F32(a), DynamicArrayHa::F32(b)) => {
            let result_op = a.add(b).unwrap();
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
        (DynamicArrayHa::F64(a), DynamicArrayHa::F64(b)) => {
            let result_op = a.add(b).unwrap();
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

fn sub_arrays(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(a), DynamicArrayHa::U8(b)) => {
            let result_op = a.sub(b).unwrap();
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
        (DynamicArrayHa::U16(a), DynamicArrayHa::U16(b)) => {
            let result_op = a.sub(b).unwrap();
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
        (DynamicArrayHa::U32(a), DynamicArrayHa::U32(b)) => {
            let result_op = a.sub(b).unwrap();
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
        (DynamicArrayHa::U64(a), DynamicArrayHa::U64(b)) => {
            let result_op = a.sub(b).unwrap();
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
        (DynamicArrayHa::F32(a), DynamicArrayHa::F32(b)) => {
            let result_op = a.sub(b).unwrap();
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
        (DynamicArrayHa::F64(a), DynamicArrayHa::F64(b)) => {
            let result_op = a.sub(b).unwrap();
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

fn mul_arrays(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(a), DynamicArrayHa::U8(b)) => {
            let result_op = a.mul(b).unwrap();
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
        (DynamicArrayHa::U16(a), DynamicArrayHa::U16(b)) => {
            let result_op = a.mul(b).unwrap();
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
        (DynamicArrayHa::U32(a), DynamicArrayHa::U32(b)) => {
            let result_op = a.mul(b).unwrap();
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
        (DynamicArrayHa::U64(a), DynamicArrayHa::U64(b)) => {
            let result_op = a.mul(b).unwrap();
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
        (DynamicArrayHa::F32(a), DynamicArrayHa::F32(b)) => {
            let result_op = a.mul(b).unwrap();
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
        (DynamicArrayHa::F64(a), DynamicArrayHa::F64(b)) => {
            let result_op = a.mul(b).unwrap();
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

fn div_arrays(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(a), DynamicArrayHa::U8(b)) => {
            let result_op = a.div(b).unwrap();
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
        (DynamicArrayHa::U16(a), DynamicArrayHa::U16(b)) => {
            let result_op = a.div(b).unwrap();
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
        (DynamicArrayHa::U32(a), DynamicArrayHa::U32(b)) => {
            let result_op = a.div(b).unwrap();
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
        (DynamicArrayHa::U64(a), DynamicArrayHa::U64(b)) => {
            let result_op = a.div(b).unwrap();
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
        (DynamicArrayHa::F32(a), DynamicArrayHa::F32(b)) => {
            let result_op = a.div(b).unwrap();
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
        (DynamicArrayHa::F64(a), DynamicArrayHa::F64(b)) => {
            let result_op = a.div(b).unwrap();
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

fn dot_arrays(
    a: DynamicArrayHa,
    b: DynamicArrayHa,
    size: usize,
    context: &Context,
) -> Result<DynamicArrayHa, &'static str> {
    match (a, b) {
        (DynamicArrayHa::U8(a), DynamicArrayHa::U8(b)) => {
            let result_op = a.matmul(b).unwrap();
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
        (DynamicArrayHa::U16(a), DynamicArrayHa::U16(b)) => {
            let result_op = a.matmul(b).unwrap();
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
        (DynamicArrayHa::U32(a), DynamicArrayHa::U32(b)) => {
            let result_op = a.matmul(b).unwrap();
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
        (DynamicArrayHa::U64(a), DynamicArrayHa::U64(b)) => {
            let result_op = a.matmul(b).unwrap();
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
        (DynamicArrayHa::F32(a), DynamicArrayHa::F32(b)) => {
            let result_op = a.matmul(b).unwrap();
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
        (DynamicArrayHa::F64(a), DynamicArrayHa::F64(b)) => {
            let result_op = a.matmul(b).unwrap();
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

pub fn ha_ndarray_test(
    data_types: Vec<&'static str>,
    operations: Vec<&'static str>,
    sizes: Vec<usize>,
) -> HashMap<&'static str, Vec<Vec<f64>>> {
    let context = Context::default().unwrap();

    let mut timings = HashMap::new();

    for op in operations {
        let mut timing_data = Vec::new();
        for dtype in &data_types {
            let mut label_data = Vec::new();
            for &size in &sizes {
                let mut results = Vec::new();
                for _ in 0..((sizes.last().unwrap() / size) + 2) {
                    let a = random_array(size, dtype, &context);
                    let b = random_array(size, dtype, &context);

                    let start;

                    let _ = match op {
                        "add" => {
                            start = Instant::now();
                            add_arrays(a, b, size, &context)
                        }
                        "sub" => {
                            let c = add_arrays(a.clone(), b, size, &context).unwrap();
                            start = Instant::now();
                            sub_arrays(c, a, size, &context)
                        }
                        "mul" => {
                            start = Instant::now();
                            mul_arrays(a, b, size, &context)
                        }
                        "div" => {
                            start = Instant::now();
                            div_arrays(a, b, size, &context)
                        }
                        "dot" => {
                            start = Instant::now();
                            dot_arrays(a, b, size, &context)
                        }
                        _ => panic!("Unsupported operation: {}", op),
                    };

                    results.push(start.elapsed().as_secs_f64());
                }
                let average_time: f64 = results.iter().sum::<f64>() / results.len() as f64;
                label_data.push(average_time.clone());

                // println!(
                //     "ha-nd-array - type: {}, size: {}, operation: {}, time: {}",
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
