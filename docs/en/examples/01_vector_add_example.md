# Vector Addition

In this section, we will use Triton to write a simple vector addition program.
Through this process, you will learn:

- The basic programming pattern of Triton.
- The `triton.jit` decorator used to define Triton kernels.

Computation kernel:

```bash
import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # Pointer to the first input vector.
               y_ptr,  # Pointer to the second input vector.
               output_ptr,  # Pointer to the output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # Note: `constexpr` will mark the variable as a constant.
               ):
    # Different data is processed by different "programs", so we need to assign:
    pid = tl.program_id(axis=0)  # Using a 1D launch grid, so the axis is 0.
    # This program will process inputs relative to the initial data offset.
    # For example, if there is a vector of length 256 with a block size of 64, programs will access elements [0:64, 64:128, 128:192, 192:256] respectively.
    # Note that offsets are a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent memory operations from accessing out of bounds.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements if the input is not a multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

Create a helper function for:

- Generating the z tensor;
- Enqueuing the above kernel with appropriate grid/block sizes.

```Python
def add(x: torch.Tensor, y: torch.Tensor):
    # Need to pre-allocate the output.
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The launch grid represents the number of kernel instances to run in parallel.
    # It can be Tuple[int] or Callable(metaparameters) -> Tuple[int].
    # In this case, use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted to a pointer to its first element.
    #  - `triton.jit` functions can be indexed by the launch grid to obtain a callable NPU kernel.
    #  - Don't forget to pass meta-parameters as keywords.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # Return the handle to z.
    return output
```

Use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

```Python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

Out:

```bash
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

"The maximum difference between torch and triton is 0.0" indicates that the output results of Triton and PyTorch are consistent.