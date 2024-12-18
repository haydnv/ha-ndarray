use std::collections::HashMap;
use std::io::{self, Write};
use std::process::Command;
use std::str;

fn execute_python_script(script: &str) -> Result<f64, String> {
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Err(String::from("Python script execution failed"));
    }

    let output_str = str::from_utf8(&output.stdout).map_err(|e| e.to_string())?;
    output_str.trim().parse::<f64>().map_err(|e| e.to_string())
}

fn max_value(size: usize, bytes: u32) -> u64 {
    let max_val_1: u64 = (2_u64.pow(bytes)) / size as u64;
    let max_val_2: u64 = bytes as u64 / 2;
    let max_val: u64 = max_val_1.min(max_val_2);
    if max_val == 0 || max_val == 1 {
        // prevents under/overflow, but maybe this should thin out
        // the array instead; cos does the compiler recognise
        // zeors in everything here?
        1
    } else {
        max_val - 1
    }
}

#[allow(unused_variables)]
pub fn py_ndarray_test(
    data_types: Vec<&'static str>,
    operations: Vec<&'static str>,
    sizes: Vec<usize>,
) -> HashMap<&'static str, Vec<Vec<f64>>> {
    let mut timings: HashMap<&str, Vec<Vec<f64>>> = HashMap::new();

    for op in operations {
        let mut timing_data = Vec::new();
        for dtype in &data_types {
            let mut label_data = Vec::new();
            for &size in &sizes {
                let mut results = Vec::new();
                for _ in 0..10 {
                    //((sizes.last().unwrap() / size) + 2) {
                    let time: f64;
                    let mut max_val: u64 = 0;
                    match *dtype {
                        "uint8" => {
                            let max_val = max_value(size, 8);
                        }
                        "uint16" => {
                            let max_val = max_value(size, 16);
                        }
                        "uint32" => {
                            let max_val = max_value(size, 32);
                        }
                        "uint64" => {
                            let max_val = max_value(size, 32);
                        }
                        _ => max_val = 1,
                    }

                    if max_val < 1 {
                        max_val = 1;
                    } else if max_val > 2 {
                        max_val = max_val - 1;
                    }

                    let _ = match op {
                        "add" => {
                            let script = format!(
                                r#"import numpy as np
import time
a = ((np.random.random(({}, {})) + 1) * {}).astype('{}')  # from 1 -> max_val
b = ((np.random.random(({}, {})) + 1) * {}).astype('{}')
start = time.perf_counter()
c = a + b
print(time.perf_counter() - start)
c = c[:, 0] + 1
"#,
                                size, size, max_val, dtype, size, size, max_val, dtype
                            );
                            time = execute_python_script(&script).unwrap();
                        }
                        "sub" => {
                            let script = format!(
                                r#"import numpy as np
import time
a = ((np.random.random(({}, {})) + 1) * {}).astype('{}')  # from 1 -> max_val
b = ((np.random.random(({}, {})) + 1) * {}).astype('{}')
c = a + b
start = time.perf_counter()
d = c - a
print(time.perf_counter() - start)
c = c[:, 0] + 1
"#,
                                size, size, max_val, dtype, size, size, max_val, dtype
                            );
                            time = execute_python_script(&script).unwrap();
                        }
                        "mul" => {
                            let script = format!(
                                r#"import numpy as np
import time
a = ((np.random.random(({}, {})) + 1) * {}).astype('{}')  # from 1 -> max_val
b = ((np.random.random(({}, {})) + 1) * {}).astype('{}')
start = time.perf_counter()
c = a * b
print(time.perf_counter() - start)
c = c[:, 0] + 1
"#,
                                size, size, max_val, dtype, size, size, max_val, dtype
                            );
                            time = execute_python_script(&script).unwrap();
                        }
                        "div" => {
                            let script = format!(
                                r#"import numpy as np
import time
a = ((np.random.random(({}, {})) + 1) * {}).astype('{}')  # from 1 -> max_val
b = ((np.random.random(({}, {})) + 1) * {}).astype('{}')
start = time.perf_counter()
c = a / b
print(time.perf_counter() - start)
c = c[:, 0] + 1
"#,
                                size, size, max_val, dtype, size, size, max_val, dtype
                            );
                            time = execute_python_script(&script).unwrap();
                        }
                        "dot" => {
                            let script = format!(
                                r#"import numpy as np
import time
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1, user_api='blas'):
    a = ((np.random.random(({}, {})) + 1) * {}).astype('{}')  # from 1 -> max_val
    b = ((np.random.random(({}, {})) + 1) * {}).astype('{}')
    start = time.perf_counter()
    c = a.dot(b)
    print(time.perf_counter() - start)
    c = c[:, 0] + 1
"#,
                                size, size, max_val, dtype, size, size, max_val, dtype
                            );
                            time = execute_python_script(&script).unwrap();
                        }
                        _ => panic!("Unsupported operation: {}", op),
                    };

                    results.push(time);
                }
                let average_time: f64 = results.iter().sum::<f64>() / results.len() as f64;
                label_data.push(average_time.clone());

                // println!(
                //     "numpy ndarray - type: {}, size: {}, operation: {}, time: {}",
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
