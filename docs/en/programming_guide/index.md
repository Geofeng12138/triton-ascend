# Triton Operator Development Guide

Overview: This article focuses on notable issues in developing Triton operators on NPUs, divided into three aspects: multi-core task parallelism, single-core data movement, and single-core data computation. First, in multi-core task parallelism, the basis for setting the maximum number of hardware cores and the specific implementation are introduced. Then, in single-core data movement, it describes in detail how to set an appropriate data block size within a loop, introduces common optimization techniques used in the process, and supplements the handling of potential UB OVERFLOW issues. Finally, returning to a single operator, it focuses on how to develop Triton operators at the single-core data computation level and emphasizes related key points.

## Document Organization

This guide separates general development principles from operator development paths categorized by hardware execution units:

- This page introduces common issues that all Triton-Ascend operators need to consider, including core division, on-chip memory, memory access, Tiling, and Autotune.
- [Vector Operator Development](./vector_operator.md) introduces operators executed primarily by the Vector Core, such as element-wise, reduction, Gather/Scatter operations.
- [Cube Operator Development](./cube_operator.md) introduces operators centered around `tl.dot`, matrix multiplication, and batched matrix multiplication.
- [CV Fusion Operator Development](./cv_fusion_operator.md) introduces scenarios where Cube computation and Vector post-processing, reduction, Softmax, or cross-core collaboration coexist within the same operator.

For simple operators, refer primarily to `docs/zh/examples/` and `third_party/ascend/tutorials/` in this repository; for complex operators, refer primarily to the complete optimization cases in `tutorial/best_practice/` on GitHub's [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops).

## General Multi-Core Task Parallelism

### Setting the Maximum Number of Hardware Cores

In a Triton operator, a grid is typically used for core division. For GPUs, the number of compute cores (SMs) is usually in the range of tens to hundreds. However, for the Ascend NPU platform, the number of compute cores (AI Cores) is on the order of tens. \
Although the runtime interface allows a maximum of 65535 concurrent tasks, tasks exceeding the number of physical cores are completed through a new round of dispatch. If Triton operators designed for GPUs are directly run on the Ascend platform, these numerous tasks will introduce significant overhead from kernel launch and initialization, affecting operator performance. \
Therefore, the core division logic needs to be modified for the Ascend platform characteristics. The most recommended approach is to **fix the number of core divisions to the physical core count of the hardware** and perform more fine-grained data tiling within the core:

* For pure Vector operators, the number of core divisions equals the **number of Vector cores**.
* For CV fusion operators, the number of core divisions equals the **number of Cube cores** (usually half the number of Vector cores). During operator execution, Vector cores are invoked in a 1:2 ratio.

Generally, on an NPU card, one compute core (AI Core) contains one Cube core, and each Cube core is paired with two Vector cores. Therefore, the number of **Vector cores (vectorcore_num)** and **Cube cores (aicore_num)** can be obtained through the following interface:

```python
import torch
import triton.runtime.driver as driver
import torch_npu

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

```

Refer to the example code: first fix the number of cores, then process task blocks in batches through an inner loop:

```python
NUM_CORE = vectorcore_num
grid = (NUM_CORE ,)
_attn_fwd[grid](Q, K, V, M, Out, acc, scale, ...)

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, scale,
              ...,
              stride_qz, stride_qh,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr
              ):
    # Calculate total tasks, flatten 3D tasks (Z, H, M) into 1D total task count
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Each core selects the tasks to process based on its own identifier
    pid = tl.program_id(0)  # Unique ID of the current core
    NUM_CORE = tl.num_programs(0)  # Get the fixed total number of launched cores
    # Loop rule: range(pid, NUM_BLOCKS, NUM_CORE) implements "strided task assignment"
    # - Start value pid: each core starts fetching tasks from its own ID to avoid overlap
    # - Step NUM_CORE: stride by the total number of cores to ensure tasks are evenly distributed
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        # Calculate data offset for each task
        # 【Core: Reverse the flattened 1D task index back to the original multi-dimensional index】
        # block_idx is the flattened 1D task index, decomposed back to original dimensions via integer division / modulo
        # 1. Decompose the combined Z+H axis & M block axis:
        #   - Integer division by NUM_BLOCKS_M: extracts the combined Z+H axis index (task_hz_idx)
        #   - Modulo by NUM_BLOCKS_M: extracts the M dimension block index (task_m_idx)
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        # 2. Decompose the combined Z+H axis into original Z and H axes:
        #   - Integer division by H: restores the Z axis index (off_z)
        #   - Modulo by H: restores the H axis index (off_h)
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        # 3. Calculate data offset: based on the restored Z/H indices, locate the starting data position in Q/K/V tensors
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
```

## General Single-Core Data Movement

### Setting an Appropriate Data Block Size (BLOCK SIZE) within a Loop

Taking `add_kernel` as an example, variables and operations together determine the on-chip memory space usage. By modifying the `BLOCK_SIZE`, the size of data blocks and intermediate computation results within the loop can be adjusted. If the upper limit is exceeded, the compiler will prompt the expected size and report an error during operator compilation. To achieve the maximum compute-to-memory-access ratio, `BLOCK_SIZE` should be as large as possible without exceeding the on-chip space. This can be achieved by pre-setting different `BLOCK_SIZE` values using Triton-Ascend's [Autotune](../examples/06_autotune_example.md); the runtime will automatically select the optimal setting.

```python
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,
               y_ptr,
               out_ptr,
               n,  # Total number of elements.
               BLOCK_SIZE: tl.constexpr,  # Number of elements per block.
               ):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(n, BLOCK_SIZE)
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        # Block size is BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        # Load x, y data to on-chip memory
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        output = x + y

        tl.store(out_ptr + offsets, output, mask=mask)
```

### Ensure the Trailing Dimension Size of Tensors is Data-Aligned as Much as Possible

[Description] For VV-type operators that require Vector core computation, the Ascend hardware's UB requires the trailing dimension size of the tensor to be divisible by 32 Bytes. For CV-type operators that require both Vector and Cube core computation, the trailing dimension size must be divisible by 512 Bytes. If the trailing dimension length is insufficient, it will be automatically padded. Under this premise, various operations on tensors with shapes like (2048, 3) and (2048, 1) in the model will suffer significant performance degradation due to automatic padding. In such cases, consider using a transpose operation to move the alignment axis to a lower dimension, and then transpose back to the original state only when storing, thus avoiding automatic padding and optimizing computation speed. Since the transpose operation itself is also subject to automatic padding rules, special techniques are also needed to avoid padding. Here is a tip called "borrow-axis transpose", suitable for scenarios where **tensor.numel() % 256Byte == 0**. The specific operation is as follows:

- Note: VV-type operators mean that the operator only uses the Vector Core during computation; CV-type operators mean that the operator uses both the AI Core and the Vector Core during computation.
- Example

```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # Load as 1D tensor. Since numel is aligned, no automatic padding occurs.
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,) # Split the long axis (2048) to borrow an alignment axis (16) for the short axis (3), making both axes aligned.
```

### First Move Data to UB, Then Select Target Values from UB

[Description] In NPU discrete scenarios, data can be first moved to UB, and then target values can be selected from the shared memory.

- Example

```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Original approach
    # val = tl.load(x_ptr + idx * stride_x, mask=mask)
    # Modified approach
    rm = tl.arange(0, M)
    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

- Performance Analysis and Comparison Before and After Optimization

By executing the use case with the msprof tool, a PROF_* folder can be obtained, which contains the op_summary_\*.csv file. This file helps analyze the pipeline status. Note: "*" represents the timestamp, [Reference method for performance data collection](../debug_guide/profiling.md).

||Op Name|aiv_mte2_time(us)|aiv_mte2_ratio|
|:---- |:--------|:--------|:--------|
|Before Optimization|pick_kernel|0.686|0.008|
|After Optimization|pick_kernel|1.041|0.066|

By analyzing the data in the table, it can be seen that the aiv_mte2_time(us) and aiv_mte2_ratio differ significantly before and after optimization. The optimization scheme first moves most of the data to the UB, reducing the number of times small batches of data are moved from L2 to UB, thereby reducing the total time for moving data from L2 to UB.

### Compute and Data Movement Overlap

Triton-Ascend supports two data processing modes: serial compute-move and overlapped compute-move.

Serial compute-move: Data is first moved from global memory to on-chip memory. After computation is complete, the next batch of data is moved. This method has significant idle waiting time and low efficiency.

Overlapped compute-move: While the first batch of data is being moved to on-chip memory, computation on it begins. Subsequently, the second batch of data is moved, forming a pipelined operation where "move + compute" overlap, significantly improving overall throughput.

The key to achieving overlapped compute-move is to design a reasonable data tiling strategy so that while the current batch of data is being computed, the data required for the next stage can be prepared in advance, thus parallelizing data movement and computation. Currently, the compiler defaults to `multiBuffer=True`, which supports overlapped compute-move by default.

### Tiling Optimization

When the AI Core performs computation, data must first be moved to on-chip memory. The on-chip memory space is usually much smaller than the total amount of data the AI Core needs to process. Taking the Atlas 800T/I A2 product as an example, the on-chip memory capacity is 192KB. When the double buffer feature is enabled by default, the capacity is halved. Therefore, operators need to tile the data during computation, loading and processing only a small portion at a time.

- Example

```diff
@libentry()
@triton.autotune(configs=runtime.get_tuned_config("masked_fill"), key=["N"])
@triton.jit
- def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
+ def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
+   base_offset = pid * BLOCK_SIZE

+   # Calculate the total number of sub-blocks to process
+   num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

+   # Loop through each sub-block
+   for sub_block_idx in range(num_sub_blocks):
+       # Calculate the offset for the current sub-block
+       sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
+       offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
-       offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        # Load input and mask
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # Write the original input first
        tl.store(out + offsets, input_vals, mask=mask)

        # Overlay and write value at the position that needs to be filled
-       value_to_write = tl.full([BLOCK_SIZE], value, dtype=input_vals.dtype)
+       value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        overwrite_vals = tl.where(fill_mask_vals, value_to_write, tl.load(out + offsets, mask=mask, other=0))
        tl.store(out + offsets, overwrite_vals, mask=mask)
```

### Triton Autotune

In Tiling optimization, the values of tiling parameters like `BLOCK_SIZE`, `BLOCK_SIZE_SUB` directly affect operator performance. However, manually debugging parameter combinations is inefficient and finding the optimal values is difficult. `triton.autotune` is an automatic tuning tool provided by the Triton framework. It can iterate through preset parameter configurations, compare performance through actual execution, and automatically select the optimal parameter combination. It is a core supporting tool for Tiling optimization.

If you are interested in the recommended usage of `configs=[]` on Triton-Ascend and the applicable boundaries of automatic Tiling, please refer further to the [Triton-Ascend autotune usage guide](../autotune_guide.md).

- Core Function
Automatically traverse the parameter space: Batch test the performance of different values for constexpr tiling parameters like `BLOCK_SIZE`, `BLOCK_SIZE_SUB`.
Performance Baseline Comparison: Use the operator's execution time as the metric to filter out the optimal parameters for the current hardware.
Cache Tuning Results: The optimal configuration after tuning is cached, and subsequent calls to the operator will reuse it directly, avoiding repeated tuning.

- Simple Example

    ```diff
    import triton.language as tl

    @triton.autotune(
    configs=[ # List of parameter configurations to test. Candidate values should be powers of 2.
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
        ],
        key=['n_elements'], # Tuning dimension: the input dimension that the parameter value depends on.
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```

- Note: Set the following environment variable to print the optimal parameter information.

    ```diff
    export TRITON_PRINT_AUTOTUNING=1
    ```

### Advanced: Using max_autotune for Automatic Tuning

For Ascend NPU operators, achieving optimal performance requires tuning not only `BLOCK_SIZE` but also multiple hardware-related parameters such as `num_stages`, `enable_hivm_auto_cv_balance`, `tile_mix_vector_loop`, etc. If `@triton.autotune` is used to manually enumerate all combinations, the configuration list would explode, making the code difficult to maintain.

`max_autotune` is an extended decorator specifically designed for the Ascend NPU (located in `triton.backends.ascend.runtime`). It allows users to provide only the basic configuration, while other tuning parameters are passed as lists. The decorator automatically generates a Config list for all combinations.

- Core Function
Developers only need to provide a small number of basic configurations (e.g., `BLOCK_SIZE`). All compiler options related to the operator type (e.g., `num_stages`, `enable_hivm_auto_cv_balance`, `tile_mix_vector_loop`, `enable_ubuf_saving`, etc.) are automatically included in the search for the optimal combination through built-in reasonable default values, without requiring explicit enumeration by the developer. This achieves automatic optimization of the optimal tiling and compiler option combination in one go. If developers want to constrain certain parameters, they can also override the default search range by explicitly passing a list.

- Simple Example

    ```diff
    from triton.backends.ascend.runtime import max_autotune

    @max_autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}),
            triton.Config({'BLOCK_SIZE': 256}),
        ],
        key=['n_elements'],
        kernel_type="vector",           # Operator type, supports cube/mix/vector
        enable_ubuf_saving=[True, False] # Optional, defaults are already included
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, **META):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    ```

### How to Avoid UB OVERFLOW on NPU

[Description] On NPU, the UB or L1 Size has an upper limit. When this error occurs, it is necessary to reduce the amount of data moved in a single operation and handle long sequence scenarios using a for loop.

```diff
E triton.compiler. errors.MLIRCompilationError:
E ///--------------------- [ERROR][Triton][BEG]-------------------------
E [ConvertLinalgRToBinary] encounters error:
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":2:1): error: Failed to run BishengHIR pipeline
E
E loc("/tmp/tmpsb6qkdih/kernel.ttadapter.mlir":3:3): error: ub overflow, requires 3072256 bits while 1572864 bits available! (possible reason
large or block number is more than what user expect due to multi-buffer feature is enabled and some ops need extra local buffer. )
```

[Note] The UB size for A2 series products is 192KB (1572864 bits).

## General Single-Core Data Computation

### Development Goal

Implement basic data computation operators (such as addition, subtraction, multiplication, division, activation functions, simple matrix element operations) on a single Ascend NPU core. Ensure efficient execution of the operator within a single core, laying the foundation for subsequent multi-core parallelism and distributed scaling.

### Development Steps

1. Determine Operator Functionality
- Clarify the shape and data type (float16/float32/int32, etc.) of input/output tensors.
- Confirm whether broadcasting or boundary handling is needed.

2. Write the Kernel Function
Single-core computation usually corresponds to block-level data processing.
Example of single-core data computation: Vector Addition

```diff

@triton.jit
def add_kernel(x_ptr, # Pointer to first input vector.
    y_ptr, # Pointer to second input vector.
    output_ptr, # Pointer to the output vector.
    n_elements, # Size of the vector.
    BLOCK_SIZE: tl.constexpr, # Number of elements each process needs to handle.
    # Note: The constexpr attribute means it can be used as a shape value.
):
    pid = tl.program_id(axis=0) # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Invocation:

 ```diff
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

Use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness.

 ```diff
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
# Out:
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')
# tensor([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], device='npu')
# The maximum difference between torch and triton is 0.0
```

3. Key Points for Single-Core Computation

- Block-level Data Processing: Each compute block is responsible for a small segment of data, ensuring parallelism.
- Boundary Check: Use `mask` or `if (tid < N)` to avoid out-of-bounds access.
- Block Size Selection: Set block and grid reasonably.

4. Performance Points:
(1) Memory Access Optimization
- Ensure contiguous access.
- Use aligned strides to avoid row/column skipping access.
- Try to align the data block size to a 32-byte boundary.
Ensure alignment when allocating input/output buffers to avoid memory access performance degradation.
Example:

 ```diff
BLOCK_SIZE = 256  # 256 * 4 bytes = 1024 bytes, well-aligned

@triton.jit
def vec_add_kernel(X, Y, Z, N,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # Calculate the index range for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask to prevent out-of-bounds access
    mask = offsets < N

    # Contiguous memory access: offsets are contiguous
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)

    z = x + y

    # Contiguous write-back
    tl.store(Z + offsets, z, mask=mask)


def vec_add(x, y):
    assert x.numel() == y.numel()
    N = x.numel()

    # Allocate aligned memory (PyTorch defaults to 64-byte alignment)
    z = torch.empty_like(x)

    # grid: each block processes BLOCK_SIZE elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    vec_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)

    return z
```

(2) Sub-block Division
- Decompose large matrices into smaller blocks, each completing computation within the UB.
- Sub-block division should balance memory access contiguity and compute unit utilization.
Example:

 ```diff
BLOCK_M = 64   # Each block processes 64 rows
BLOCK_N = 64   # Each block processes 64 columns
BLOCK_K = 32   # Inner accumulation dimension

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)  # Block id in the M direction
    pid_n = tl.program_id(1)  # Block id in the N direction

    # Starting coordinates for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over tiled blocks
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0
        )
        b = tl.load(
            B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0
        )
        acc += tl.dot(a, b)

    # Write back result
    c = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```


## General Multi-Dimensional Tensor Tiling

When Triton operators process multi-dimensional tensors, the core idea is to map high-dimensional data to hardware Blocks, Cores, and hardware units. This section provides typical processing examples for 2D and 3D tensors.

### 2D Tensor Tiling: Taking Matrix Multiplication (GEMM) as an Example

For 2D matrix multiplication, it is usually necessary to perform 2D tiling on the height (M) and width (N), and iterate over the depth (K) in a loop.

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 1. Task Division: Calculate the coordinates of the current Block on the M and N dimensions
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. Define Block Pointers, handling multi-dimensional strides
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # 3. Loop over the K dimension for accumulation
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_bn[None, :] < N))
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn, accumulator)
```

**Key Points**:

- `pid_m` / `pid_n` correspond to the block indices on the M / N dimensions, respectively.

- `stride_*` explicitly handles multi-dimensional strides, avoiding assumptions about contiguous memory.

- The K dimension is accumulated by looping over tiled blocks.

### 3D and Higher-Dimensional Tensor Tiling: Taking Batched GEMM as an Example

When processing 3D tensors (e.g., `[Batch, M, N]`), the `Batch` dimension (B) can be directly mapped to a Triton `Grid` dimension, or flattened with the `M/N` dimensions and then remapped.

#### Adding the `Batch` Dimension to the Launch `Grid`

```python
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), B)
```

#### Kernel Function Implementation

```python
@triton.jit
def batched_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, B, ...):
    # Get the index of the current Batch
    pid_b = tl.program_id(2)

    # Calculate the base address offset in global memory based on the Batch index
    a_batch_ptr = a_ptr + pid_b * M * K
    b_batch_ptr = b_ptr + pid_b * K * N
    c_batch_ptr = c_ptr + pid_b * M * N

    # The subsequent tiling of M, N, K dimensions is exactly the same as 2D GEMM, just replace the base address pointers
    # ...
```

**Key Points**:

- `tl.program_id(2)` gets the index of the Batch dimension.

- Each Batch independently calculates its own `a_batch_ptr` / `b_batch_ptr` / `c_batch_ptr`.

- The subsequent tiling logic for M / N / K dimensions is consistent with 2D GEMM.