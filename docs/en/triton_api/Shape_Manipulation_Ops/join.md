# triton.language.join

## 1 Function Description

Concatenates two input tensors of the same shape along a new innermost dimension. The output tensor has one more dimension than the input tensors, with a size of 2, while keeping other dimensions unchanged.

**Syntax:**

- `triton.language.join(x, y)` - Function call form
- `x.join(y)` - Member function form

**Functionality:**

- Concatenates two input tensors of the same shape along a new innermost dimension
- The output tensor has one more dimension than the input tensors, with a size of 2
- Keeps other dimensions unchanged

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| x | tensor | Yes | First input tensor |
| y | tensor | Yes | Second input tensor |

**Return Value:**

- **Type:** tensor
- **Shape:** The broadcasted shape of the input tensors plus a dimension of size 2
- **Data Type:** Same as the input tensors
- **Memory Layout:** Stacks x and y along the new dimension

**Constraints:**

- Both input tensors must have shapes and data types that can be broadcast to the same shape

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|---------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Constraints

None

### 2.5 Usage Example

```python
import torch
import triton
import triton.language as tl

@triton.jit
def join_example(out_ptr):
    # Create two 2x3 tensors
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.full([2, 3], 1.0, dtype=tl.float32)

    # Concatenate, resulting in 2x2x3
    z = tl.join(x, y)

    # Write the result back to the external tensor
    offs = (
        tl.arange(0, 2)[:, None, None] * (2 * 3)
        + tl.arange(0, 2)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, z)

## Example call
out = torch.empty((2, 2, 3), dtype=torch.float32, device="npu")
join_example[(1,)](out)
print(out.shape)  # Output: torch.Size([2, 2, 3])
```