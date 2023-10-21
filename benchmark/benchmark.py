'''
benchmark utility comparing numpy and ndarray, with the ha-ndarray
'''
# std
import time

# 3rd
import numpy as np
from matplotlib import pyplot as plt

# local
from ha_ndarray import ndarray_test
from ha_ndarray import ha_ndarray_test


if __name__ == "__main__":

    data_types = ['uint8', 'uint16', 'uint32', 'uint64', 'float32', 'float64']
    data_types_2 = ['u8', 'u16', 'u32', 'u64', 'f32', 'f64']
    sizes = [2**1, 2**3, 2**5, 2**7, 2**9, 2**11]#, 2**13]#, 2**15]
    operation = ['add', 'sub', 'mul', 'div', 'dot']

    # dictionary to times
    timings_py = {
        'add': np.zeros((len(data_types), len(sizes))),
        'sub': np.zeros((len(data_types), len(sizes))),
        'mul': np.zeros((len(data_types), len(sizes))),
        'div': np.zeros((len(data_types), len(sizes))),
        'dot': np.zeros((len(data_types), len(sizes)))
    }

    for op in operation:  # Skip 'add' as it's already done
        for k in range(len(data_types)):
            for j in range(len(sizes)):
                results = []
                print(f'Py - type: {data_types[k]}, size: {sizes[j]}, operation: {op}')
                for replicates in range((sizes[-1] // sizes[j]) + 2):
                    a = np.random.rand(sizes[j], sizes[j]).astype(data_types[k])
                    b = np.random.rand(sizes[j], sizes[j]).astype(data_types[k])
                    b[b == 0] = 1
                    start = time.perf_counter()
                    if op == 'sub':
                        c = a - b
                    elif op == 'mul':
                        c = a * b
                    elif op == 'div':
                        c = a / b
                    elif op == 'dot':
                        c = np.dot(a, b)
                    elif op == 'add':
                        c = a + b
                    else: # throw shade
                        raise ValueError(f'op: {op} not supported')
                    results.append(time.perf_counter() - start)

                timings_py[op][k, j] = np.array(results).mean()


    timings_ndarray = ndarray_test()
    timings_ndarray = {k: np.array(v) for k, v in timings_ndarray.items()}

    # i havent learnt how to use your crate yet
    timings_ha_array = ha_ndarray_test()
    timings_ha_array = {k: np.array(v) for k, v in timings_ha_array.items()}

    # Plotting results for each operation
    print(f'len data_types: {len(data_types)}, len operation: {len(operation)}')
    fig, axes = plt.subplots(len(data_types), len(operation), figsize=(15, 10))

    lines = ['-', '--', '-.', ':', '-']

    sizes_sq = [sq ** 2 for sq in sizes]

    for i, _ in enumerate(data_types):
        for j, op in enumerate(operation):
            if len(data_types) == 1 and len(operation) == 1:
                ax = axes
            elif len(operation) == 1:
                ax = axes[i]
            else:
                ax = axes[i, j]
            ax.set_xlabel('size (# elements 2D)')
            ax.set_ylabel('time (s)')
            ax.set_title(f'op: {op}; datatype: {data_types[i]}')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True)

            # plotting
            ax.plot(sizes_sq, timings_py[op][i, :], label=f'np', color=[1, 0, 1])
            ax.plot(sizes_sq, timings_ndarray[op][i, :], label=f'nd', color=[1, 0.5, 0])
            ax.plot(sizes_sq, timings_ha_array[op][i, :], label=f'ha', color=[0, 0.5, 1.0])

    axes[i, j].legend()

    plt.tight_layout()
    plt.savefig('fig.png')
    plt.show()
