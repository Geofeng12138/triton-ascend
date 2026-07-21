# Accuracy Comparison

In this section, we will use Triton to write a simple accuracy comparison program.
Through this process, users will learn:

- Accuracy comparison methods for each Triton data type.
- Reference example code: triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py

Computation kernel:

```Python
def test_add(x0, x1):
    """
    Test whether the vector addition implemented by Triton matches the PyTorch result in accuracy comparison.

    Steps:
    1. Compute the reference result using PyTorch (torch_ref)
    2. Write a Triton kernel and compute the result (triton_cal)
    3. Call accuracy_comparison to perform accuracy comparison
    """

    # 1. Use PyTorch as the reference implementation (golden truth)
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    # 2. Define the Triton kernel (executed on NPU/GPU)
    @triton.jit
    def triton_kernel_add(
        out_ptr0,   # Output pointer: location to store the result
        in_ptr0,    # Input pointer 0: starting address of x0
        in_ptr1,    # Input pointer 1: starting address of x1
        XS: tl.constexpr  # constexpr parameter: vector length, determined at compile time
    ):
        # Generate an index array [0, 1, 2, ..., XS-1]
        idx = tl.arange(0, XS)
        # Load the value of x0 from in_ptr0 + idx
        tmp0 = tl.load(in_ptr0 + idx)
        # Load the value of x1 from in_ptr1 + idx
        tmp1 = tl.load(in_ptr1 + idx)
        # Perform addition
        tmp2 = tmp0 + tmp1
        # Store the result to out_ptr0 + idx
        tl.store(out_ptr0 + idx, tmp2)

    # 3. Triton wrapper function: calls the kernel and returns the result
    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)  # Create an output tensor with the same shape and dtype as input
        # Launch kernel: grid = [1, 1, 1] means using only one block
        # Note: XS must be passed as an argument because it is of type tl.constexpr
        triton_kernel_add[1, 1, 1](y0, x0, x1, XS=x0.numel())
        return y0

    # 4. Obtain the reference result and the Triton computed result
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)

    # 5. Accuracy comparison
    accuracy_comparison(triton_cal, torch_ref)

    # 6. Print success message
    print(f"== dtype:{triton_cal.dtype} == The accuracy comparison between triton_cal and torch_ref was successful.")


```

Create an accuracy comparison function that adapts to each dtype, using the corresponding accuracy comparison method.

```Python

def accuracy_comparison(y_cal, y_ref):
    """
    Accuracy comparison function: selects the appropriate comparison strategy based on data type.

    Handling strategies for different data types:
    - Floating point types (float16/32, bfloat16): use torch.testing.assert_close, set relative/absolute error tolerances
    - Integer types (int8/16/32/64): require exact equality (torch.equal)
    - Boolean type (bool): strict comparison on CPU (to avoid device differences)
    """
    # Check that the output data types match
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # Move tensors to NPU (assuming the test is performed on NPU)
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # Select different comparison methods based on data type
    if tensor_dtype == torch.float16:
        # float16 has lower precision, allow slightly larger errors
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16 has even lower precision, it is recommended to convert to float32 before comparison
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # float32 has higher precision, use stricter tolerances
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # Integer types should be exactly equal
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # Boolean types should be compared on CPU to avoid device-specific boolean representation differences
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


```

You can use the following command to run the reference example code: tutorials/14-accuracy-comparison.py

```Python
python triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py
```