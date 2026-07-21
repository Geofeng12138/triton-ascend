# Migrating GPU Triton Operators

Overview: This document describes the general approach and common issues when migrating GPU Triton operators to Ascend NPU. During migration, it is recommended to first replace the Python-side device and runtime interfaces, then check grid partitioning, memory access alignment, single-core computation, UB space, and coreDim limitations. Finally, complete code modification and correctness verification using specific examples.

## General Migration Process

### Migrating Python-side Device and Runtime Interfaces

Before modifying the specific Triton kernel, first complete the Python-side device migration:

1. Add `import torch_npu` to the Python file.
2. Find device specifications like `device="cuda"`, `device='cuda'`, `.cuda()`, and `.to("cuda")`, and change them to `device="npu"`, `device='npu'`, `.npu()`, or `.to("npu")`.
3. Find GPU-specific interfaces like `torch.cuda.*`, CUDA stream, CUDA event, CUDA synchronize, and replace them with NPU counterparts or remove unnecessary synchronization logic.
4. Remove logic that only serves GPU device discovery, such as device assertions related to `triton.runtime.driver.active.get_active_torch_device()`.
5. Keep the main logic of the Triton kernel unchanged, first compile and verify correctness using NPU tensors.

### Adjusting Grid Partitioning

Common practice on GPUs is to design the grid as a large number of logical programs, scheduled by hardware and runtime to execute on SMs. When migrating to NPU, priority should be given to the physical number of AI Cores and the operator type:

- Prefer 1D grids; 2D NPU adaptation will also merge into 1D, e.g., `(20,)` has the same effect as `(4, 5)`.
- The number of concurrent tasks for vector-only operators is usually organized by the number of Vector Cores; operators containing `tl.dot` are usually organized by the number of AI Cores.
- When the logical grid is much larger than the number of physical cores, evaluate whether to change to processing multiple tiles in a loop within each program, or use `TRITON_ALL_BLOCKS_PARALLEL` when there is no sequential dependency between logical cores.
- coreDim cannot exceed `UINT16_MAX` (65535). For large shape operators, control the grid size by adjusting `BLOCK_SIZE` or the tiling method.

| Dimension | Core Architecture | Operator Type |
|-----------|------------------|---------------|
| Ascend NPU | Multiple AI Cores, divided into Cube Core (matrix multiplication) and Vector Core (vector computation) | Vector-only operators → Number of concurrent tasks = Number of Vector Cores; Operators with `tl.dot` → Number of concurrent tasks = Number of AI Cores |
| GPU NVIDIA/AMD | Multiple CUDA Cores (scalar/vector computation) + Tensor Cores (matrix multiplication) | GPU operators generally have concurrency determined automatically by the compiler and hardware |

### Checking Single-Core Data Movement

After completing the device replacement, continue to check the data movement method within a single program:

- Vector operator scenarios require 32-byte memory access alignment; cube-vector fusion operator scenarios require 512-byte alignment.
- Keep tail masks to ensure boundary elements do not access out of bounds.
- Check the on-chip memory usage of one tile to avoid triggering UB space overflow.
- Remove or replace GPU-specific synchronization APIs, such as CUDA thread, stream, event, or kernel synchronize related interfaces.

### Checking Single-Core Data Computation

NPU and GPU differ in their computation units and supported data types. After migration, first ensure correctness, then adjust based on performance issues:

- For intermediate values like integer indices, offsets, and lengths, first confirm whether the current data type is efficiently supported by the NPU path.
- For operators containing `tl.dot`, confirm whether the M/N/K tiles, accumulation dtype, and output dtype meet the NPU backend requirements.
- For long sequences, long hidden sizes, or large K-dimension loops, prioritize controlling the size of each load and computation through tiling.

## Migration Examples

### Example 1: Complete Vector Addition Migration

```diff
import torch
+import torch_npu  # [Added] Import Ascend NPU PyTorch adaptation library to provide NPU device support
import triton
import triton.language as tl

-DEVICE = triton.runtime.driver.active.get_active_torch_device()  # [Deleted] GPU device auto-detection, not needed for NPU

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input vector.
    y_ptr,  # Pointer to second input vector.
    output_ptr,  # Pointer to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
-    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE  # [Deleted] GPU device consistency check, not needed for NPU
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

torch.manual_seed(0)
size = 98432
-x = torch.rand(size, device='cuda')  # [Deleted] GPU device specification
+x = torch.rand(size, device='npu')  # [Modified] Specify Ascend NPU device
-y = torch.rand(size, device='cuda')  # [Deleted] GPU device specification
+y = torch.rand(size, device='npu')  # [Modified] Specify Ascend NPU device
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)
```

### Example 2: Device Replacement and Single-Core Data Movement

The following example demonstrates correctness verification for a single-core data movement scenario after replacing the device from CUDA to NPU:

```diff
import pytest
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def fn_broadcast_1d(output_ptr, x_ptr, XS: tl.constexpr, YS: tl.constexpr):
    xidx = tl.arange(0, XS)[None, :]
    base = tl.load(x_ptr + xidx)
    out = base.broadcast_to((YS, XS))
    oidx = tl.arange(0, YS)[:, None] * XS + tl.arange(0, XS)[None, :]
    tl.store(output_ptr + oidx, out)

@pytest.mark.parametrize('shape', [(1,), (2,), (4,)])
@pytest.mark.parametrize('dtype', [torch.int32])
def test_npu_1d(shape, dtype):
    XS = shape[0]
    YS = 4

-    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='cuda')
+    x = torch.randint(-1000, 1000, (XS,), dtype=dtype, device='npu')
    std = torch.broadcast_to(x, (YS, XS))
-    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='cuda')
+    output = torch.randint(-1000, 1000, (YS, XS), dtype=dtype, device='npu')
    fn_broadcast_1d[(1,)](output, x, XS, YS)
    assert torch.allclose(std, output)
```

## Common Issues Overview

After completing the basic migration steps, new issues may arise. These can be categorized into two types:

1. coreDim Limitation Issue

   Triggered when the grid dimension exceeds NPU hardware limits.
   Typical error message: `coreDim=xxxx can't be greater than UINT16_MAX`.

2. UB Space Overflow

   Memory usage exceeds NPU cache capacity.
   Typical error message: `ub overflow, requires xxxx bits while 1572864 bits available!`.

### Resolving coreDim Exceeding Limit

Problem Analysis:
The NPU coreDim parameter cannot exceed UINT16_MAX (65535). When processing large-scale data, simple grid partitioning may break this limit.

Case: zeros_like Function Optimization
Data size: N = 1073741824, original BLOCK_SIZE = 2048, calculated coreDim = 524288 > 65535 (exceeds limit)

Solution 1:
The Ascend compiler has a corresponding solution for coreDim exceeding the limit. Simply set the environment variable 'TRITON_ALL_BLOCKS_PARALLEL' to 1. The command is as follows:

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

Solution 2:
Increase BLOCK_SIZE to reduce the number of required cores, ensuring coreDim does not exceed the limit.
The calculation formula is as follows:

```text
coreDim = ceil(N / BLOCK_SIZE)
ceil(N / BLOCK_SIZE) <= 65535
BLOCK_SIZE >= ceil(N / 65535)
```

Substituting `N = 1073741824` yields:

```text
ceil(1073741824 / 65535) = 16385
triton.next_power_of_2(16385) = 32768
```

Therefore, if `BLOCK_SIZE` is a power of 2, it should be at least `32768`.

Code before optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=1024)  # Original value too small
    return out
```

Code after optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)
@triton.jit
def zeros_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 0.0, mask=mask)

def zeros_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    logger.debug("GEMS ZEROS_LIKE")
    if device is None:
        device = x.device # x.device = "npu"
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(x, device=device, dtype=dtype)
    N = x.numel()
    min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
    BLOCK_SIZE = max(32768, min_block_size) # At least 32768
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    zeros_kernel[grid_fn](out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out
```

### Dynamically Calculating Suitable BLOCK_SIZE to Avoid coreDim Exceeding Limit

```diff
optimal_block_size = 32768  # Optimized value based on calculation

grid_fn = lambda meta: (triton.cdiv(N, optimal_block_size),)

zeros_kernel[grid_fn](out, N, BLOCK_SIZE=optimal_block_size)
return out
```

### Handling Compound Issues: coreDim + UB Overflow

Problem Analysis:
In some cases, solving the coreDim issue may trigger a new UB overflow issue. This usually occurs when increasing BLOCK_SIZE causes the amount of data a single thread block needs to process to exceed the NPU's UB cache capacity.

Case:
Data size: N = 1073741824, original BLOCK_SIZE = 4096, calculated coreDim = 262144 > 65535 (exceeds limit), adjusted to BLOCK_SIZE = 32768, coreDim = 32768 (compliant), but UB overflow occurs.

Solution:
Introduce the BLOCK_SIZE_SUB parameter to further subdivide large blocks, controlling memory usage while maintaining a reasonable coreDim.
Code before optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)
    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)

def masked_fill(inp, mask, value):
    # ... parameter validation code ...
    # inp.device = "npu"
    out = torch.zeros_like(inp)
    N = inp.numel()
    if N == 0:
        return out

    grid = lambda meta: (triton.cdiv(N, 4096),)  # Causes coreDim to exceed limit
    masked_fill_kernel[grid](inp, mask.to(torch.int), value, out, N, 4096)
    return out
```

Code after optimization:

```diff
import logging
import torch
import torch_npu
import triton
import triton.language as tl
logger = logging.getLogger(__name__)

@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
    BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    # Calculate the number of sub-blocks to process
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    # Process in blocks to avoid UB overflow
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        # Load and process data in batches
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        # First write original data
        tl.store(out + offsets, input_vals, mask=mask)
        # Then overwrite target values at positions needing fill
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)

def masked_fill(inp, expand_mask, value):
    logger.debug("GEMS MASKED FILL")

    out = torch.zeros_like(inp)
    # ... parameter validation code ...
    # inp.device = "npu"
    N = inp.numel()
    if N == 0:
        return out

    # Use optimized parameter configuration
    MAIN_BLOCK_SIZE = 32768  # Ensure coreDim compliance
    SUB_BLOCK_SIZE = 1024    # Control UB usage

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, expand_mask.to(torch.int), value, out, N,
                        MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

### Why Does UBSIZE Exceed Memory Error Occur

Unreasonable partitioning leads to excessive unaligned memory access or computation. For example, moving 2D data of (64, 32) with stride (12832, 128). If it were aligned data access, the stride would be (32, 1). For unaligned access content, add an axis of size 1 to the innermost axis, making it (64, 32, 4). Since the hardware requires 32-byte alignment for UB memory in vector operator scenarios, assuming type=float16, the corresponding stride should be (12832, 128, 1).

### Line-by-Line Comparison of Dispersed Memory Access Code to Observe Scalar Inefficient Mapping

Set the environment variable `TRITON_DEBUG=1` to save `~/.triton/cache/xxx.ttadapter`, then execute:

```diff
bishengir-compile xxx.ttadapter --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --hivm-compile-args=bishengir-print-ir-after=hivm-inject-sync
```

This will output IR. Compare the Triton operator logic with the operations inside the IR to observe if any operations are not mapped to instructions.
Observe if there is pure scalar movement or computation in the HIVM IR stage that is not mapped to SIMD instructions, as this will become a performance bottleneck.

Issue: Dispersed Memory Access && Scalar Inefficient Mapping
`b[1024, 32] = a[1024, 32]` The original Triton approach uses threads to bind the lowest dimension 32 of [1024, 32] to thread blocks, then divides 1024 by 16 to get [64, 16, 32], and then binds 64 to thread blocks.

```diff
chunk_fwd_kernel_o[(NT, B * H)](
    p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
    block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(1024,), # One-dimensional tensor
        strides=(32,), # Contiguous memory
        offsets=(i_t * 16,), # Start from the beginning
        block_shape=(BT,), # Block size
        order=(0,) # Contiguous access
    )
​)
```

Optimization Idea

Adjust the shape/stride of block_ptr:
Treat (1024, 32) as a 2D matrix. The lowest dimension 32 is contiguous, so the stride should be (32, 1), not (32,). This way, each thread block can access 32 contiguous elements. Bind the thread block to the row dimension (1024), and each thread processes a whole row of 32 elements. This makes memory access contiguous and has good affinity.

For example:

```diff
block_ptr = tl.make_block_ptr(
    base=input_ptr,
    shape=(1024, 32),
    strides=(32, 1),
    offsets=(i_t * BT, 0),
    block_shape=(BT, 32),
    order=(1, 0) # Row-major layout: dimension 1 is most contiguous (stride 1), dimension 0 is least contiguous
)
```