# Matrix Multiplication

In this section, we demonstrate a kernel implementation for matrix multiplication using Triton.

## Computation Kernel

The following Triton kernel implements a Batched Matrix Multiplication with Bias:
The computation formula is:
$$ \text{output}[b, i, j] = \sum_k \text{x}[b, i, k] \cdot \text{y}[k, j] + \text{z}[b, i, j] $$
where:

- `x` has shape `(A, B)`
- `y` has shape `(B, C)`
- `z` (bias) has shape `(A, C)`
- The output `output` has shape `(A, C)`

This kernel assumes that a single block is responsible for computing the entire output matrix, making it suitable for small-scale matrices (where A, B, C are small enough to be fully covered by the current program block).

```python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_dot_2_Bias(
    output_ptr,   # Output tensor pointer, shape (A, C)
    x_ptr,        # Input tensor x pointer, shape (A, B)
    y_ptr,        # Input tensor y pointer, shape (B, C)
    z_ptr,        # Bias tensor z pointer, shape (A, C)
    A: tl.constexpr,  # First dimension size (batch / number of rows)
    B: tl.constexpr,  # Shared dimension (columns of x, rows of y)
    C: tl.constexpr   # Second dimension size (number of columns)
):
    # Create index vectors
    bidx = tl.arange(0, A)  # [0, 1, ..., A-1], used for row dimension
    cidx = tl.arange(0, B)  # [0, 1, ..., B-1], used for columns of x / rows of y
    didx = tl.arange(0, C)  # [0, 1, ..., C-1], used for column dimension

    # Construct linear indices for x: (A, B) -> flattened to A*B
    Xidx = bidx[:, None] * B + cidx[None, :]  # Broadcast to form (A, B) index grid

    # Construct linear indices for y: (B, C) -> flattened to B*C
    Yidx = cidx[:, None] * C + didx[None, :]  # (B, C) index grid

    # Construct linear indices for z and output: (A, C)
    Zidx = bidx[:, None] * C + didx[None, :]  # (A, C) index grid

    # Load data from global memory
    X = tl.load(x_ptr + Xidx)  # Load (A, B) sub-block
    Y = tl.load(y_ptr + Yidx)  # Load (B, C) sub-block
    Z = tl.load(z_ptr + Zidx)  # Load bias (A, C)

    # Perform matrix multiplication and add bias
    ret = tl.dot(X, Y) + Z  # tl.dot performs (A, B) × (B, C) → (A, C)

    # Write result back to global memory
    oidx = bidx[:, None] * C + didx[None, :]  # Same as Zidx, can be reused
    tl.store(output_ptr + oidx, ret)
```

## Utility Methods

The following helper functions support testing and verification of the Triton kernel, including a PyTorch reference implementation, data type mapping, random tensor generation, and result validation.

```Python
def torch_dot_Bias(x0, x1, bias):
    """PyTorch reference implementation: performs matrix multiplication and adds a bias term."""
    res = torch.matmul(x0, x1) + bias
    return res

def get_torch_typename(dtype):
    """Maps a string-form data type to the corresponding torch.dtype."""
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

def generate_tensor(shape, dtype):
    """Generates a random tensor based on the specified shape and data type, adapting to different numerical type ranges."""
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref):
    """Compares Triton computation results with PyTorch reference results on NPU, using tolerances or strict equality based on data type."""
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
```

## Parameterized Tests

Use `pytest` to perform parameterized functional validation of the `triton_dot_2_Bias` kernel, covering different matrix dimension and data type combinations.

```python
# Test case configuration: (A, B, C) represents matrix x: (A,B), y: (B,C), bias/output: (A,C)
testlist = [
    (16, 16, 16),
]

# List of supported data types (currently only float16)
typelist = ['float16',]

@pytest.mark.parametrize('A, B, C', testlist)
@pytest.mark.parametrize('sigtype', typelist)
def test_dot_2_Bias(sigtype, A, B, C):
    """End-to-end functional test for the triton_dot_2_Bias kernel."""
    dtype = get_torch_typename(sigtype)

    # Generate input tensors and move to NPU
    x0 = generate_tensor(shape=(A, B), dtype=sigtype).npu()
    x1 = generate_tensor(shape=(B, C), dtype=sigtype).npu()

    # Bias term is generated using float32 uniformly (to avoid precision issues with integer bias)
    if 'int' in sigtype:
        bias = generate_tensor(shape=(A, C), dtype='int32').npu()
        # Integer inputs need to be converted to float32 for computation, then converted back to target type
        ans = torch_dot_Bias(x0.to(torch.float32), x1.to(torch.float32), bias.to(torch.float32)).to(dtype)
    else:
        bias = generate_tensor(shape=(A, C), dtype='float32').npu()
        ans = torch_dot_Bias(x0, x1, bias).to(eval(f"torch.{dtype}"))

    # Initialize output tensor
    output = torch.zeros((A, C), dtype=dtype).npu()

    # Launch Triton kernel (grid=(1,1,1), single block execution)
    triton_dot_2_Bias[1, 1, 1](output, x0, x1, bias, A, B, C, debug=True)

    # Verify result correctness
    validate_cmp(sigtype, output, ans)
    print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C}) PASSED!")


if __name__ == "__main__":
    # Supports running a single test case directly (convenient for debugging)
    test_dot_2_Bias("float16", 16, 16, 16)
```

**Output Example:**

```python
Test matmul with dtype=float16, shape=(16,16,16) PASSED!
```

The output log above indicates that the results from Triton and PyTorch are completely consistent.