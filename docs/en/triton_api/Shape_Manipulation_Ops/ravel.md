# triton.language.ravel

## 1 Function Description

Flattens the input tensor into a one-dimensional tensor, preserving the order of elements in memory. The total number of elements in the output tensor is the same as the input tensor.

**Syntax:**

- `triton.language.ravel(input)` - Function call form
- `input.ravel()` - Member function form

**Functionality:**

- Flattens the input tensor into a one-dimensional tensor
- Preserves the order of elements in memory
- The total number of elements in the output tensor is the same as the input tensor

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |

**Return Value:**

- **Type:** tensor
- **Shape:** One-dimensional tensor containing all elements of the input tensor
- **Data Type:** Same as the input tensor
- **Memory Layout:** Flattened in row-major order

**Constraints:**

- No special constraints; supports input of any shape

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
@triton.jit
def flatten_kernel(x_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    # Load 2D data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Flatten to one dimension
    x_flat = x.ravel()

    # Store flattened result
    tl.store(output_ptr + offsets, x_flat, mask=mask)
```