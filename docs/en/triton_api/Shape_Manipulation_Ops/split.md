# triton.language.split

## 1 Function Description

Splits the input tensor into two tensors along the last dimension. The last dimension size of the output tensors is half of the input tensor's last dimension, while other dimensions remain unchanged.

**Syntax:**

- `triton.language.split(input)` - Function call form
- `input.split()` - Member function form

**Functionality:**

- Splits the input tensor into two tensors along the last dimension
- The last dimension size of the output tensors is half of the input tensor's last dimension; the last dimension size must be 2
- Other dimensions remain unchanged

## 2 Parameter Specification

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |

**Return Value:**

- **Type:** Tuple[tensor, tensor]
- **Shape:** Two tensors with the same shape, the last dimension is half of the input
- **Data Type:** Same as the input tensor
- **Memory Layout:** Contains the elements at odd and even positions of the input tensor respectively

**Constraints:**

- The last dimension size of the input tensor must be even
- Outputs two tensors with the same shape

### 2.2 DataType Support Table

| Supported | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|-----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Constraints

None

### 2.5 Usage Example

```python
@triton.jit
def complex_split_kernel(complex_ptr, real_ptr, imag_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Load complex data
    complex_data = tl.load(complex_ptr + offsets, mask=mask)

    # Split into real and imaginary parts
    real_part, imag_part = complex_data.split()

    # Store real and imaginary parts
    tl.store(real_ptr + offsets, real_part, mask=mask)
    tl.store(imag_ptr + offsets, imag_part, mask=mask)
```