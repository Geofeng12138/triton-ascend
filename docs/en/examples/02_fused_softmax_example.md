# Fused Softmax

In this section, we will write a fused softmax operation program using Triton.
Along the way, you will learn:

- The advantages of kernel fusion for bandwidth-bound operations.
- Reduction operations in Triton.

## Naive Softmax Computation on X Row-wise Using Native PyTorch

```Python
import torch
import torch_npu

import triton
import triton.language as tl

def naive_softmax(x):
    """
    We subtract the maximum element to avoid overflow. Softmax is invariant to this shift.
    """
    # Read MN elements; write M elements
    x_max = x.max(dim=1)[0]
    # Read MN + M elements; write MN elements
    z = x - x_max[:, None]
    # Read MN elements; write MN elements
    numerator = torch.exp(z)
    # Read MN elements; write M elements
    denominator = numerator.sum(dim=1)
    # Read MN + M elements; write MN elements
    ret = numerator / denominator[:, None]
    # Total: read 5MN + 2M elements; write 3MN + 2M elements
    return ret
```

Purpose of Kernel Fusion

When implemented natively in PyTorch, computing `y=naive_softmax(x)` requires reading 5MN+2M elements from DRAM and writing back 3MN+2M elements. This is clearly very inefficient; we would prefer to use a custom "fused" kernel that reads x only once and performs all necessary computations on-chip.
This would only require reading and writing back 2MN bytes, so we can expect a theoretical speedup of approximately 4x (i.e., (8MN+4M)/2MN).

`torch.jit.script` is designed to automatically perform this kind of "kernel fusion," but it is still far from ideal.

## Computation Kernel

The softmax kernel works as follows: each program loads a set of rows from the input matrix X with a stride equal to the number of programs, performs normalization, and writes the result to the output matrix Y.
Note: An important limitation of Triton is that each block must have a power-of-two number of elements. Therefore, if we want to handle arbitrary input shapes, we need to internally "pad" each row and ensure correct memory operations.

```Python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Program start row
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride indicates how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit
        # the row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask because BLOCK_SIZE may be greater than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract the maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that the exponential operation in Triton is fast but approximate.
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write the output back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

We can create a helper function that enqueues the kernel and its meta-parameters to handle any given input tensor.

```Python
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape

    # The block size for each loop iteration is the smallest power of two greater than or equal to the number of columns of `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Allocate output space
    y = torch.empty_like(x)

    # Pre-compile the kernel to get register usage and compute occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        num_programs = 32
        kernel = softmax_kernel
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE
    )
    return y
```

## Unit Test

We need to test the processed kernel on a matrix with irregular numbers of rows and columns to verify that the padding mechanism works.

```Python
torch.manual_seed(0)
x = torch.randn(1823, 781, device='npu')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
print(y_triton)
print(y_torch)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(y_triton-y_torch))}')
```

Out:

```bash
tensor([[0.0002, 0.0017, 0.0009,  ..., 0.0009, 0.0013, 0.0073],
        [0.0001, 0.0004, 0.0006,  ..., 0.0006, 0.0004, 0.0003],
        [0.0007, 0.0002, 0.0006,  ..., 0.0011, 0.0004, 0.0039],
        ...,
        [0.0021, 0.0002, 0.0015,  ..., 0.0012, 0.0014, 0.0022],
        [0.0003, 0.0002, 0.0007,  ..., 0.0005, 0.0006, 0.0007],
        [0.0034, 0.0014, 0.0005,  ..., 0.0007, 0.0016, 0.0028]],
       device='npu:0')
tensor([[0.0002, 0.0017, 0.0009,  ..., 0.0009, 0.0013, 0.0073],
        [0.0001, 0.0004, 0.0006,  ..., 0.0006, 0.0004, 0.0003],
        [0.0007, 0.0002, 0.0006,  ..., 0.0011, 0.0004, 0.0039],
        ...,
        [0.0021, 0.0002, 0.0015,  ..., 0.0012, 0.0014, 0.0022],
        [0.0003, 0.0002, 0.0007,  ..., 0.0005, 0.0006, 0.0007],
        [0.0034, 0.0014, 0.0005,  ..., 0.0007, 0.0016, 0.0028]],
       device='npu:0')
The maximum difference between torch and triton is 1.4901161193847656e-08
```

"The maximum difference between torch and triton is 1.4901161193847656e-08" indicates that the output results of Triton and PyTorch are very close and indistinguishable to the naked eye.