mod benchmark_ha;
mod benchmark_nd;
mod benchmark_py;
use std::collections::HashMap;

fn print_table(
    data_types: &[&str],
    operations: &[&str],
    sizes: &[usize],
    timings_py: &HashMap<&str, Vec<Vec<f64>>>,
    timings_nd: &HashMap<&str, Vec<Vec<f64>>>,
    timings_ha: &HashMap<&str, Vec<Vec<f64>>>,
) {
    println!(
        "\n\n\n|------------|-----------|------|-----------|-----------|-----------|-------|-------|"
    );
    println!(
        "| {:<10} | {:<9} | {:<4} | {:<9} | {:<9} | {:<9} | {:<5} | {:<5} |",
        "Operation", "Data Type", "Size", "Py Timing", "ND Timing", "HA Timing", "Py/HA", "ND/HA"
    );
    println!(
        "|------------|-----------|------|-----------|-----------|-----------|-------|-------|"
    );

    for &op in operations {
        for &dtype in data_types {
            for &size in sizes {
                let py_time = timings_py[op][dtype_index(dtype)][size_index(size, sizes)];
                let nd_time = timings_nd[op][dtype_index(dtype)][size_index(size, sizes)];
                let ha_time = timings_ha[op][dtype_index(dtype)][size_index(size, sizes)];

                let py_ha = py_time / ha_time;
                let nd_ha = nd_time / ha_time;

                println!(
                    "| {:<10} | {:<9} | {:<4} | {:<9.7} | {:<9.7} | {:<9.7} | {:<5.2} | {:<5.2} |",
                    op, dtype, size, py_time, nd_time, ha_time, py_ha, nd_ha
                );
            }
        }
    }

    println!(
        "|------------|-----------|------|-----------|-----------|-----------|-------|-------|"
    );
}

fn dtype_index(dtype: &str) -> usize {
    match dtype {
        "uint8" => 0,
        "uint16" => 1,
        "uint32" => 2,
        "uint64" => 3,
        "float32" => 4,
        "float64" => 5,
        _ => unreachable!(),
    }
}

fn size_index(size: usize, sizes: &[usize]) -> usize {
    sizes.iter().position(|&s| s == size).unwrap()
}

fn main() {
    println!("Hi");

    let size_max: u32 = 9;

    let data_types: Vec<&str> = vec!["uint8", "uint16", "uint32", "uint64", "float32", "float64"];
    let operations: Vec<&str> = vec!["add", "sub", "mul", "div", "dot"];
    let sizes: Vec<usize> = (1..=size_max).step_by(2).map(|x| 2_usize.pow(x)).collect();

    let timings_py: HashMap<&str, Vec<Vec<f64>>> =
        benchmark_py::py_ndarray_test(data_types.clone(), operations.clone(), sizes.clone());

    let timings_nd: HashMap<&str, Vec<Vec<f64>>> =
        benchmark_nd::ndarray_test(data_types.clone(), operations.clone(), sizes.clone());

    let timings_ha: HashMap<&str, Vec<Vec<f64>>> =
        benchmark_ha::ha_ndarray_test(data_types.clone(), operations.clone(), sizes.clone());

    print_table(
        &data_types,
        &operations,
        &sizes,
        &timings_py,
        &timings_nd,
        &timings_ha,
    );
}
