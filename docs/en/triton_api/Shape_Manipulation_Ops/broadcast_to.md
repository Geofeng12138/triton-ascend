# triton.language.broadcast_to

## 1 Function Description

Broadcasts a tensor to a target shape, automatically handling dimension alignment. The broadcast operation does not copy data; instead, it achieves this by changing the tensor's shape and strides.

**Syntax:**

- `triton.language.broadcast_to(input, shape)` - Function call form
- `input.broadcast_to(shape)` - Member function form

**Functionality:**

- Automatically handles dimension alignment, expanding dimensions of size 1 to match the corresponding dimension size in the target shape.
- Keeps the data unchanged, only modifying the tensor's shape information.

## 2 Parameter Specifications

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| shape | List[int] | Yes | Target shape |

**Return Value:**

- **Type:** tensor
- **Shape:** Same as the target shape specified by the `shape` parameter
- **Data Type:** Same as the input tensor
- **Memory Layout:** Broadcast is achieved by modifying stride information; no data copy occurs.

**Constraints:**

- The number of dimensions of the input tensor must equal the number of dimensions of the target shape.
- All dimensions must satisfy the broadcasting rules.

### 2.2 DataType Support Table

| Support Status | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Constraints

Unlike `broadcast`, the `broadcast_to` implemented by the Triton community requires that the rank of the tensor's shape and the target shape must be the same.

### 2.5 Usage

**Basic Usage:**

```python
@triton.jit
def matrix_add_bias_kernel(x_ptr, bias_ptr, output_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Load data block
    x = tl.load(x_ptr + offsets, mask=mask)

    # Broadcast bias to match the shape
    bias = tl.load(bias_ptr)
    bias_broadcast = bias.broadcast_to([BLOCK_M, BLOCK_N])

    # Perform addition
    output = x + bias_broadcast
    tl.store(output_ptr + offsets, output, mask=mask)
```