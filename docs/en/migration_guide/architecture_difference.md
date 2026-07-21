# Differences Between Ascend and GPU Development

## Multi-Core Task Parallelism Strategy

On NPUs, Triton multi-core parallelism uses a physical core strong-binding mode, which fundamentally differs from the GPU's logical dimension parallelism combined with hardware automatic physical mapping. The core comparison is shown in the table below:

| Dimension | GPU (NVIDIA) | Ascend NPU |
|-----------|--------------|------------|
| Grid Essence | Logical task dimension (decoupled from physical cores) | Physical core group mapping (bound to AI Core topology) |
| Core Count / Dimension Limit | No hard limit on grid dimension/size | Grid size ≤ total AI Core count, 2D must match topology |

GPUs can bind multiple dimension axes (a 3D grid=[n,m,l] is equivalent to n×m×l parallel threads), where each thread corresponds to a single kernel execution that runs only once. \
NPUs have multiple physical cores (Vector cores, Cube cores), with different hardware generations having different core counts. Each core executes a Block only once and supports repeated scheduling execution of that Block.

### Fully Utilizing Core Count

Ascend NPUs have multiple compute cores. Reasonably allocating and fully utilizing all available cores is a key factor in improving operator performance.
When calling a Triton kernel function, the number of cores used is controlled by setting launch parameters. Taking the GELU operator as an example:

```Python
triton_gelu[n, 1, 1](...)  # The first parameter indicates the number of cores used, n means using n cores
```

By tuning the core count, full scheduling and utilization of all compute resources can be achieved, thereby maximizing parallelism and throughput. When `auto-blockify` (see next section) is not enabled, the number of cores in the emitted grid must be less than or equal to 65,535.

### auto-blockify: Breaking the 65,535 Logical Block Limit

Community Triton on NVIDIA GPUs treats the grid as a purely logical dimension — `n` logical blocks are mapped 1:1 to `n` hardware blocks. At runtime, the hardware distributes them to SMs, and each block does not require an internal loop. On Ascend, due to the physical core strong-binding described in the previous section, the upper limit of the launchable grid is stuck at 65,535, which is too restrictive for kernels with millions of logical work items (such as autotuned reduce/scan, megablocks-style sparse kernels, etc.).

`auto-blockify` (the `SIMTAutoBlockify` compile-time pass + the corresponding runtime cap) eliminates this limitation by "treating it as logical at compile time and folding it into physical cores at launch time":

- **Compile Time**: The Triton pass wraps the kernel function body in a `scf.for` loop, where the iteration variable is provided by `gpu.linear_block_id`. The chunk size = `ceildiv(logical_block_count, physical_core_count)`, and each physical block sequentially runs `chunk` logical block IDs.
- **Runtime**: The block-count parameter passed to the launcher is clamped from the logical grid to `physical_core_count`, consistent with the compile-time folding.

Both sides share the same gating metadata (`enable_auto_blockify` on `NPUOptions`, falling back to `TRITON_ALL_BLOCKS_PARALLEL` when not set). The compile-time loop wrapping and the runtime cap are always synchronized — there is no scenario where a kernel is compiled under one mode but launched under another.

Precautions when porting from GPU Triton kernels:

- Grids larger than 65,535 can run directly without manually folding the outer dimension into the kernel function body.
- Logical blocks must remain order-independent (the loop accesses chunks sequentially). Kernels that rely on strict logical block ID order (e.g., cross-block synchronization based on a specific order) need to be rewritten.
- Per-block workspace allocation drops from `O(logical_block_count)` to `O(physical_core_count)` because the workspace is reused across iterations of the inner `scf.for` loop.

## Single-Core Data Movement Strategy

### Data Tiling

When writing Triton kernel functions, a reasonable data tiling strategy is crucial for performance optimization. By adjusting different tiling granularity parameters, the computational load and memory access efficiency can be balanced across different dimensions.

Common tiling parameters include:

```text
ncore: Number of cores used (cross-core tiling)
xblock: Data block size between cores (inter-core tiling)
xblock_sub: Intra-core tiling granularity (fine-grained intra-core partitioning)
```

Developers can manually select the optimal tiling configuration based on the actual scenario, ensuring that each computation fully utilizes on-chip memory and avoids performance bottlenecks caused by frequent access to global memory.

Taking the GELU operator as an example, by adjusting the tiling parameters, the on-chip cache capacity limit can be effectively adapted, thereby improving execution efficiency.

Note: The on-chip memory capacity of the Atlas 800T/I A2 product is 192KB. Therefore, this limitation must be considered when designing the tiling strategy to ensure that the data volume per computation round does not exceed the on-chip memory capacity.

#### GELU Operator Example

GELU operator development example, using 3 methods to compute the result.

`standard_unary` is the standard Torch computation.

`triton_easy_kernel` is a simple Triton implementation.

`triton_better_kernel` is a more efficient Triton implementation.

#### Standard Torch Implementation

Input tensor x0, compute the GELU operator via torch computation, and return the result value.

```Python
def standard_unary(x0):
    res = x0 * 0.5 * (1.0 + torch.erf(x0 / torch.sqrt(torch.tensor(2.0))))
    return res
```

#### Simple Triton Implementation

The following is a simple kernel example written in Triton, demonstrating how to define and call a basic Triton kernel function. This example implements a simple mathematical operation (GELU activation function).

```Python
# Define the triton_kernel kernel function
@triton.jit
def triton_easy_kernel(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    tl.store(out_ptr0 + idx_block, ret)
```

Precautions

1. Memory Limitation: In the above implementation, all input data is loaded into memory at once for computation. If the input tensor is too large, it may exceed the on-chip memory capacity of a single kernel, leading to an out-of-memory error.
Therefore, this simple implementation is more suitable for small-scale tensor computations or for understanding the basic writing and calling methods of Triton kernels.

2. Applicable Scenarios: Although this method helps quickly understand and get started with Triton programming, for large-scale datasets or high-performance application scenarios, it is recommended to adopt more complex data tiling strategies (such as Tiling) to fully utilize hardware resources and avoid memory overflow issues. Through this approach, developers can quickly get started with Triton programming while understanding how to define, call, and optimize Triton kernel functions.

#### More Efficient Triton Implementation

When writing high-performance operators using Triton on Ascend NPUs, to fully utilize hardware resources, avoid memory overflow, and improve execution efficiency, a data tiling strategy is typically required. Below is an optimized Triton kernel implementation example suitable for large-scale tensor computations.

```Python
# Define the triton_kernel kernel function
@triton.jit
def triton_better_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = x_index < xnumel
        x = tl.load(in_ptr0 + x_index, xmask)
        ret = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
        tl.store(out_ptr0 + x_index, ret, xmask)

# Call the triton_kernel kernel function
ncore = 32
xblock = 32768
xblock_sub = 8192
triton_better_kernel[ncore, 1, 1](x0, out1, x0.numel(), xblock, xblock_sub)
```

Key Code Explanation

```Python
# Calculate the starting offset address of the data block processed by the current core, achieving inter-core tiling. Each core is only responsible for a data range of size XBLOCK.
xoffset = tl.program_id(0) * XBLOCK

# Further subdivide the data block within a single core, processing data of size XBLOCK_SUB each time, achieving intra-core tiling.
for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):

# Construct the data index array for the current iteration, used to access input and output tensors.
x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]

# Set a mask to prevent out-of-bounds access, ensuring only data within the valid range is processed.
xmask = x_index < xnumel

# Used to load data from global memory to on-chip memory and write computation results back to global memory, respectively.
tl.load() and tl.store()
```

## Compilation Optimization Capabilities

### AscendNPU IR Optimization

Targeting the characteristics of Ascend software and hardware, compilation options for AscendNPU IR optimization have been adapted, as shown in the table below.
**Usage**: Pass the value of the compilation option during the autotune configuration phase.
Taking enabling the `multibuffer` option as an example, during the autotune configuration phase, i.e., in `triton.Config`, pass `'multibuffer': True`. See the [autotune example](../examples/06_autotune_example.md) for details:

```python
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),]
```

| Option | Capability | Enabled |
| ----------------- | ------------ | ----------------- |
| multibuffer | Enables pipelined data movement | Default true; true, false. Configurable in autotune. |
| unit_flag | An optimization item for Cube output | Default None; true, false. Configurable in autotune. |
| limit_auto_multi_buffer_only_for_local_buffer | An optimization item for CV operators, an optimization item for Cube output | Default None; true, false. Configurable in autotune. |
| limit_auto_multi_buffer_of_local_buffer | Specific scope for enabling double buffer in Cube operators | Default None; can be "no-limit" or "no-l0c". Configurable in autotune. |
| set_workspace_multibuffer | Configures the workspace multi-buffer level, used to enable multi-buffering for workspace-related data movement. | Default None; can take a single value, e.g., 2 or 4; configurable candidate values in autotune. |
| enable_hivm_auto_cv_balance | Enables or disables automatic CV balance, used to balance Cube and Vector execution in CV fusion scenarios. | Default None; true, false. Configurable in autotune. |
| tile_mix_vector_loop | An optimization item for CV operators, indicating how many parts the current Vector can be split into. | Default None; can take a single value, e.g., 2, 4, or 8; configurable candidate values in autotune. |
| tile_mix_cube_loop | An optimization item for CV operators, indicating how many parts the current Cube can be split into. | Default None; can take a single value, e.g., 2, 4, or 8; configurable candidate values in autotune. |
| auto_blockify_size | Optimization item for `TRITON_ALL_BLOCKS_PARALLEL`, used to specify the size of the expanded leftmost dimension. | Default 1; can take a single integer value, e.g., 2, 4, or 8; configurable candidate values in autotune. |
| enable_auto_blockify | Per-kernel level override of the `TRITON_ALL_BLOCKS_PARALLEL` environment variable. When explicitly set to **true** or **false**, the kernel takes effect according to this value (ignoring the environment variable); when not set (None), it is determined by the environment variable. Priority: This option > Environment variable > Off. Both the compile-time blockify pass and the runtime block-count cap take effect according to this resolved value, ensuring they are always consistent. | Default None; can be **true** / **false** / None. |

- Note: The optimization compilation options are in the `ascend/backend/compiler.py` code.
- Note: A CV operator indicates that the operator uses both the AI Core and the Vector Core during its computation.