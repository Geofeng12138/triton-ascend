# Autotune

If you wish to first understand the recommended usage of Triton-Ascend autotune, the meaning of `configs=[]`, and the applicable boundaries of automatic Tiling, it is recommended to read the [Triton-Ascend autotune usage guide](../autotune_guide.md).

In this section, we will demonstrate using Triton's autotune method to automatically select the optimal kernel configuration parameters. The current Triton-Ascend autotune is fully compatible with the community autotune usage (refer to the [community documentation](https://triton-lang.org/main/python-api/generated/triton.autotune.html)), meaning users need to manually pass in a set of predefined `triton.Config` objects, and then autotune selects the optimal kernel configuration through benchmarking. Additionally, Triton-Ascend provides **advanced autotune** usage, where users do not need to provide information such as the kernel's split axes or tiling axes. The autotune automatically parses these axes based on the triton kernel semantics, automatically generates some potentially optimal kernel configurations, and then selects the best configuration through benchmarking or profiling.

Note:
Currently, Triton-Ascend autotune supports block size and multibuffer (compiler optimization). Due to hardware architecture differences, it does not support `num_warps` and `num_stages` parameters. More tunable items for autotune will be added in the future.

## Community Autotune Usage Example

```Python
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():

    # Returns a set of different kernel configurations for autotune testing
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]

    @triton.autotune(
        configs=get_autotune_config(),      # Configuration list
        key=["numel"],                      # Triggers autotune when numel size changes
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # Block size, controls how much data each thread block processes
    ):
        pid = tl.program_id(0)            # Get the current program ID
        idx = pid * XS + tl.arange(0, XS) # Index range processed by the current thread block
        msk = idx < numel                 # Mask to avoid out-of-bounds access

        # Repeat some computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # Load x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # Load x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # Compute
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # Store to output

    # Triton calling function, automatically uses the autotuned kernel
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # Calculate grid size
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # Compare with PyTorch reference implementation
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"                         # Use NPU as device
    DTYPE = torch.float32
    N = 192 * 1024                      # Input length
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # Random input x0
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # Random input x1
    torch_ref = torch_calc_func(x0, x1)              # Get reference result
    triton_cal = triton_calc_func(x0, x1)            # Run Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # Verify output consistency

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # Print success message
```

## Advanced Autotune Usage Example

```Python
# The following explains the key parameter usage differences between advanced autotune and the community version
#
# configs:
# - Community autotune (default) requires explicitly passing a set of triton.Config objects. The framework compiles and benchmarks each configuration to select the optimal one.
# - Advanced autotune: The framework automatically generates candidate tiling configurations based on the kernel, then compiles and benchmarks each configuration to select the optimal one.
# * Note: 1. To enable advanced mode, users must manually import triton.backends.ascend.runtime;
#         2. If configs=[], the framework automatically generates candidate tiling configurations based on the kernel. Note that in this case, the @triton.autotune decorator must be applied directly above @triton.jit,
#            with no other decorators (e.g., libentry) in between;
#         3. If configs is not empty, the framework will not automatically generate candidate tiling configurations by default;
#         4. If configs is not empty and hints.auto_gen_config=True, the framework automatically generates Configs and merges them with user-defined Configs for configuration selection;
#         5. The advanced version supports setting the performance collection method via os.environ["TRITON_BENCH_METHOD"] = ( "npu" ).
#
# hints(Dict[str, str]):
# Note: 1. hints is optional. When not provided by the user, the framework automatically parses relevant parameters like split_params and tiling_params.
#       2. Users can pass parameters via hints to generate tiling. This involves split_params, tiling_params, low_dim_axes, and reduction_axes, and all four parameters must be provided together.

# split_params (Dict[str, str]): A dictionary of axis name: argument name. The argument is a tunable parameter for the split axis, e.g., 'XBLOCK'
#     The axis name must be in the set of axis names for the parameter key. Do not add the prefix r to the axis name.
#     This parameter can be empty. When both split_params and tiling_params are empty, automatic optimization will not occur.
#     The split axis can usually be determined by the `tl.program_id()` kernel launch statement.
# tiling_params (Dict[str, str]): A dictionary of axis name: argument name. The argument is a tunable parameter for the tiling axis, e.g., 'XBLOCK_SUB'
#     The axis name must be in the set of axis names for the parameter key. Do not add the prefix r to the axis name.
#     This parameter can be empty. When both split_params and tiling_params are empty, automatic optimization will not occur.
#     The tiling axis can usually be determined by the `tl.arange()` tiling expression.
# low_dim_axes (List[str]): A list of axis names for all low-dimensional axes. The axis name must be in the set of axis names for the parameter key.
# reduction_axes (List[str]): A list of axis names for all reduction axes. The axis name must be in the set of axis names for the parameter key. Add the prefix r to the axis name.
# auto_gen_config (bool): Defaults to False. Involves the following scenario combinations:
#     1. If the user does not define Config, regardless of whether auto_gen_config is set, the framework automatically generates Config by default;
#     2. If the user defines Config and auto_gen_config=False, the framework does not automatically generate Config and only uses the user-defined Config;
#     3. If the user defines Config and auto_gen_config=True, the framework automatically generates Config and merges it with the user-defined Config for configuration selection;
#
# key (list[str]/Dict[str,str]):
# - Passes a list of runtime parameter names; a change in any parameter value in the list triggers regeneration and evaluation of candidate configurations.
# Note: 1. If hints passes information for split_params, tiling_params, low_dim_axes, and reduction_axes, the key type must be Dict[str,str], as in Example 1:
#       2. If hints does not pass information for split_params, tiling_params, low_dim_axes, and reduction_axes, the key type must be list[str], and axis information will be assigned according to parameter order, as in Example 2:

Example 1:
@triton.autotune(
    configs=[],
    key={"x":"n_elements"},
    hints={
        "split_params":{"x":"BLOCK_SIZE"},
        "tiling_params":{},
        "low_dim_axes":["x"],
        "reduction_axes":[],
    }
)
Example 2:
@triton.autotune(
    configs=[],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input vector.
    y_ptr,  # Pointer to the second input vector.
    output_ptr,  # Pointer to the output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each kernel should process.
    # Note: `constexpr` indicates it can be determined at compile time, so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D grid, so the axis is 0.
    # The offset of the data that the current kernel will process in memory relative to the starting address.
    # For example, if you have a vector of length 256 and a block_size of 64, the programs
    # will access elements [0:64, 64:128, 128:192, 192:256] respectively.
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent memory operations from accessing out-of-bounds elements.
    mask = offsets < n_elements
    # Load x and y, masking out extra elements in case the input vector length is not a multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write back x + y.
    tl.store(output_ptr + offsets, output, mask=mask)
```

Note:

1. Triton-Ascend defaults to using benchmarking to measure on-chip computation time. When the environment variable `export TRITON_BENCH_METHOD="npu"` is set, it uses `torch_npu.profiler.profile` to obtain the on-chip computation time for each kernel configuration. For triton kernels with fast computation, such as small shape operators, this can provide more accurate computation time compared to the default method, but it will significantly increase the overall autotune time. Please enable with caution.
2. Currently, this advanced usage targets Vector-type operators and does not support Cube-type operators. For more advanced usage examples, please refer to the [autotune advanced usage examples](https://gitcode.com/Ascend/triton-ascend/tree/main/third_party/ascend/unittest/autotune_ut/).

### Automatic Parameter Parsing

Before performing automatic parameter parsing, the system first obtains the parameters that were not passed when the `kernel` function was called. **These unpassed parameters are considered candidates for split axis and tiling axis parameters.**

```Python
@triton.jit
def kernel_func(
    outputptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    # kernel implementation
    ...

# XBLOCK and XBLOCK_SUB are not passed, so they are candidates for split and tiling axis parameters
# BLOCK_SIZE is passed as a keyword argument, so it is not a candidate parameter and will not be identified
kernel_func[grid](y, x, n_rows, n_cols, BLOCK_SIZE=block_size)
```

#### Split Axis Parameter Parsing

Split axis parameter parsing is determined based on the `tl.program_id()` kernel launch statement. The system identifies potential split axis parameters by analyzing the usage of `tl.program_id()` variables and their multiplication operations with other variables (currently supports direct multiplication or indirect multiplication through intermediate variables), and filters them based on the candidate parameter list (parameters not provided by the user).

Finally, the specific split axis corresponding to the parameter is confirmed through mask comparison and the `key` passed in `autotune`.

Note: 1. The split axis parameter must be multiplied by `tl.program_id()`. 2. A mask comparison must be performed, and the key corresponding to this axis must be used directly as an rvalue or as the parameter of a min function acting as an rvalue to correspond to the specific split axis; otherwise, parameter parsing will fail. 3. The identified split axis parameters are limited to the candidate parameter list, ensuring that only parameters that can be dynamically adjusted through autotune are considered.

```Python
@triton.autotune(
    configs=[],
    key={"n_elements"} # Must be specified
    ...
)
@triton.jit
def triton_func(...):
    # case1:
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)

    # case2:
    block_start = tl.program_id(0) * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)

    # case3:
    offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)

    # mask compare
    mask = offsets < n_elements # 1
    mask = offsets < min(..., n_elements) # 2

# Parsed split axis parameter: split_params = {"x": "XBLOCK"}
```

#### Tiling Axis Parameter Parsing

Tiling axis parameter parsing is determined based on `tl.arange()`, `tl.range()`, and `range()` tiling statements. By analyzing the usage of `tl.range()`, `tl.arange()`, and `range()` in `for` loops within the program and the variables they compute, potential tiling axis parameters are identified. Common parameters of `tl.range()` or `range()` and `tl.arange()` are extracted, and filtering is performed based on the candidate parameter list (parameters not provided by the user).

Finally, the specific tiling axis corresponding to the parameter is confirmed through mask comparison and the `key` passed in `autotune`.

Note: 1. The tiling axis parameter must appear in the call to `tl.arange()` and must participate in the calculation of the loop range within a `for` loop via `tl.range()`, `range()`, or integer division (`//`). 2. A mask comparison must be performed, and the key corresponding to this axis must be used directly as an rvalue or as the parameter of a min function acting as an rvalue to correspond to the specific tiling axis; otherwise, parameter parsing will fail. 3. The identified tiling axis parameters are limited to the candidate parameter list, ensuring that only parameters that can be dynamically adjusted through autotune are considered.

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # Must be specified
    ...
)
@triton.jit
def triton_func(...):
    ...
    # case 1
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

    # case 2
    loops = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(loops):
        row_offsets = loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        ...
        xmask = row_offsets < n_rows # 1
        xmask = row_offsets < min(..., n_rows) # 2
        ymask = col_offsets < n_cols

# Parsed tiling axis parameter: tiling_params = {"x": "XBLOCK_SUB"}
# Although BLOCK_SIZE is also in tl.arange and compared with n_cols to compute a mask, it is not a tiling axis parameter.
```

#### Low-Dimensional Axis Parameter Parsing

Low-dimensional axis parameter parsing is determined based on `tl.arange()` tiling statements. By analyzing the usage of `tl.arange()` in the program and the variables it computes, potential low-dimensional axis parameters are identified. The `tl.arange()` itself and the variables it participates in computing are extracted. Filtering is performed by checking whether slicing operations are used for dimension expansion and by judging the expanded dimensions.

Finally, the low-dimensional axis of the current kernel is confirmed through mask comparison and the `key` passed in `autotune`.

Note: 1. The low-dimensional axis must be computed via `tl.arange()` and undergo slicing. It must be expanded in a non-lowest dimension or not participate in slicing to be identified. 2. If no mask comparison is performed, it cannot correspond to a specific low-dimensional axis, leading to parameter parsing failure.

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # Will be automatically assigned in order as {"x": "n_rows", "y": "n_cols"}
    ...
)
@triton.jit
def triton_func(...):
    ...
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols

# Parsed low-dimensional axis: low_dim_axes = {"y"}
# Although row_offsets is also computed via tl.arange and compared with n_rows to compute a mask, the slicing is expanded in the low dimension, so x is not a low-dimensional axis.
```

#### Pointer Parameter Parsing

Pointer-type parameter parsing is determined based on whether the parameter participates in memory access statements like `tl.load()` and `tl.store()`.

First, all parameters of the kernel function are parsed. Then, all variables involved in the computation for each parameter are recursively found.

If the parameter directly participates, or if an intermediate variable computed from the parameter indirectly participates, in the calculation of the first argument of `tl.load()` or `tl.store()`, the parameter is considered a pointer-type parameter.

Note: 1. Variables modified with `tl.constexpr` are not pointer-type variables and will not be parsed further. 2. Only memory access statements where the parameter directly participates or where an intermediate variable computed from the parameter (after one level of computation) indirectly participates are considered. Intermediate variables resulting from two or more levels of computation from the parameter are not counted.

```Python
@triton.autotune(...)
@triton.jit
def triton_func(input_ptr, output_ptr, ...):
    ...
    # case1
    input = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input, mask=mask)

    # case2
    inputs_ptr = input_ptr + offsets
    input = tl.load(inputs_ptr, mask=mask)
    outputs_ptr = output_ptr + offsets
    tl.store(outputs_ptr, input, mask=mask)

# Parsed pointer-type parameters: input_ptr, output_ptr
```

## More Features

### Profiling Results for Automatically Generated Optimal Configuration

```Python
# Automatically generates profiling results for the current autotune optimal kernel configuration in the `auto_profile_dir` directory, i.e., performance data collected using `torch_npu.profiler.profile`
# This works in both community autotune usage and advanced autotune usage
@triton.autotune(
    auto_profile_dir="./profile_result",
    ...
)
```