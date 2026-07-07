# triton.language.permute

## 1 Function Description

Reorders the dimensions of a tensor according to the `dims` parameter. This operation does not change the tensor's data, only the order of its dimensions. Supports reordering of any number of dimensions.

**Syntax:**

- `triton.language.permute(input, dims)` - Function call form
- `input.permute(dims)` - Member function form

**Functionality:**

- Reorders the dimensions of a tensor according to the `dims` parameter
- Does not change the tensor's data, only the order of its dimensions
- Supports reordering of any number of dimensions

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| dims | List[int] | Yes | New dimension order |

**Return Value:**

- **Type:** tensor
- **Shape:** Dimensions reordered according to the `dims` parameter
- **Data Type:** Same as the input tensor
- **Memory Layout:** Transposition achieved by modifying stride information; no data copy occurs

**Constraints:**

- `dims` must contain all dimension indices of the input tensor

### 2.2 DataType Support Table

| Supported | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|-----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Constraints

* Transposition of tensors with more than 8 dimensions is not supported

### 2.5 Usage Example

```python
import torch
import triton
import triton.language as tl

@triton.jit
def permute_example(out_ptr):
    # Create a 2x3x4 tensor
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # Transpose dimensions to become 4x2x3
    y = tl.permute(x, [2, 0, 1])

    # Write the result back to the external tensor
    offs = (
        tl.arange(0, 4)[:, None, None] * (2 * 3)
        + tl.arange(0, 2)[None, :, None] * 3
        + tl.arange(0, 3)[None, None, :]
    )
    tl.store(out_ptr + offs, y)

## Call example
out = torch.empty((4, 2, 3), dtype=torch.float32, device="npu")
permute_example[(1,)](out)
print(out.shape)  # Output: torch.Size([4, 2, 3])
```