# triton.language.trans

## 1 Function Description

Transposes the dimensions of a tensor according to the `dims` parameter. It does not change the tensor's data, only the order of the dimensions. This is a specially optimized transpose operation.

**Syntax:**

- `triton.language.trans(input, dims)` - Function call form
- `input.trans(dims)` - Member function form

**Functionality:**

- Transposes the dimensions of a tensor according to the `dims` parameter
- Does not change the tensor's data, only the order of the dimensions
- A specially optimized transpose operation

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| dims | List[int] | Yes | Dimension order after transposition |

**Return Value:**

- **Type:** tensor
- **Shape:** Dimensions rearranged according to the `dims` parameter
- **Data Type:** Same as the input tensor
- **Memory Layout:** Transposition is achieved by modifying stride information; no data copy is performed

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
import triton
import triton.language as tl

@triton.jit
def trans_example():
    # Create a 2x3x4 tensor
    x = tl.zeros([2, 3, 4], dtype=tl.float32)

    # Transpose dimensions to 4x2x3
    y = tl.trans(x, [2, 0, 1])

    return y

## Call example
result = trans_example()
print(result.shape)  # Output: (4, 2, 3)
```