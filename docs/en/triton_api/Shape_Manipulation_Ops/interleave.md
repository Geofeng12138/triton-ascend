# triton.language.interleave

## 1 Function Description

Interleaves two input tensors of the same shape along the last dimension. The last dimension of the output tensor is twice the size of the input tensors, while other dimensions remain unchanged.

**Syntax:**

- `triton.language.interleave(x, y)` - Function call form
- `x.interleave(y)` - Member function form

**Functionality:**

- Interleaves two input tensors of the same shape along the last dimension
- The last dimension of the output tensor is twice the size of the input tensors
- Other dimensions remain unchanged

## 2 Parameter Specification

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| x | tensor | Yes | First input tensor |
| y | tensor | Yes | Second input tensor, must have the same shape as x |

**Return Value:**

- **Type:** tensor
- **Shape:** Last dimension of the input shape multiplied by 2
- **Data Type:** Same as the input tensors
- **Memory Layout:** Alternating elements of x and y

**Constraints:**

- Both input tensors must have the same shape and data type
- The shape of the output tensor is the input shape with the last dimension multiplied by 2

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|---------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Restrictions

None

### 2.5 Usage Example

```python
import triton
import triton.language as tl

@triton.jit
def interleave_example():
    # Create two 2x3 tensors
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.ones([2, 3], dtype=tl.float32)

    # Interleave, resulting in 2x6
    z = tl.interleave(x, y)

    return z

## Call example
result = interleave_example()
print(result.shape)  # Output: (2, 6)
```