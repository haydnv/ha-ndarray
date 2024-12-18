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

fn random_array(size: usize, datatype: &str, context: &Context, op: &str) -> DynamicArrayHa {
    let mut rng = rand::thread_rng();
    let array: Vec<f32> = (0..size * size).map(|_| rng.gen_range(0.0..=1.0)).collect();
    let mapped_array: Vec<f32>;
    // map array; 95% of number == 1, 5% of numbers are 2 (avoids over/underflows)
    let threshold: f32 = (((size * size) - size) / (size * size)) as f32;
    if op == "div" {
        mapped_array = array
            .iter()
            .map(|x| if x > &threshold { 2.0 } else { 1.0 })
            .collect();
    } else {
        mapped_array = array
            .iter()
            .map(|x| if x > &threshold { 0.0 } else { 1.0 })
            .collect();
    }
    match datatype {
        "uint8" => DynamicArrayHa::U8(
            ArrayBase::<Vec<_>>::with_context(
                context.clone(),
                vec![size, size],
                mapped_array.iter().map(|x| *x as u8).collect(),
            )
            .unwrap(),
        ),
        "uint16" => DynamicArrayHa::U16(
            ArrayBase::<Vec<_>>::with_context(
                context.clone(),
                vec![size, size],
                mapped_array.iter().map(|x| *x as u16).collect(),
            )
            .unwrap(),
        ),
        "uint32" => DynamicArrayHa::U32(
            ArrayBase::<Vec<_>>::with_context(
                context.clone(),
                vec![size, size],
                mapped_array.iter().map(|x| *x as u32).collect(),
            )
            .unwrap(),
        ),

        "uint64" => DynamicArrayHa::U64(
            ArrayBase::<Vec<_>>::with_context(
                context.clone(),
                vec![size, size],
                mapped_array.iter().map(|x| *x as u64).collect(),
            )
            .unwrap(),
        ),
        "float32" => DynamicArrayHa::F32(
            ArrayBase::<Vec<_>>::with_context(
                context.clone(),
                vec![size, size],
                mapped_array.iter().map(|x| *x as f32).collect(),
            )
            .unwrap(),
        ),
        "float64" => DynamicArrayHa::F64(
            ArrayBase::<Vec<_>>::with_context(
                context.clone(),
                vec![size, size],
                mapped_array.iter().map(|x| *x as f64).collect(),
            )
            .unwrap(),
        ),
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
                for _ in 0..10 {
                    //((sizes.last().unwrap() / size) + 2) {
                    let a = random_array(size, dtype, &context, op);
                    let b = random_array(size, dtype, &context, op);

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
