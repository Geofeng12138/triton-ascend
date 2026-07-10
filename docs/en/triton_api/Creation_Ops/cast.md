# triton.language.cast

## 1 Function Description

Converts a tensor to a specified data type, supporting numeric type conversion, bit-level reinterpretation (bitcast), floating-point downcast rounding modes, and Ascend extended integer overflow handling modes.

**Syntax:**

- `triton.language.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)` - Function call form
- `input.cast(dtype, fp_downcast_rounding=None, bitcast=False)` - Member function form

**Functionality:**

- Numeric type conversion: integer <-> integer, floating-point <-> floating-point, integer <-> floating-point
- Bit-level reinterpretation (bitcast): does not change bits, only changes the interpretation type
- Floating-point downcast supports rounding modes: `rtne` (default, round to nearest even), `rtz` (toward zero)
- Integer conversion (Ascend extension) supports overflow modes: `trunc` (truncation, default), `saturate` (saturation)

## 2 Parameter Specification

### 2.1 Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| input | tensor | Yes | Input tensor |
| dtype | tl.dtype | Yes | Target data type |
| fp_downcast_rounding | str | No | Only valid for floating-point downcast, `rtne` or `rtz` |
| bitcast | bool | No | Whether to perform bit-level reinterpretation, default False |
| overflow_mode | str | No | Ascend extension: integer overflow handling, `trunc` or `saturate` |

**Return Value:**

- **Type:** tensor
- **Shape:** Same as the input tensor
- **Data Type:** Same as the target type specified by the dtype parameter
- **Memory Layout:** Determined by the bitcast parameter whether to perform bit-level reinterpretation

**Constraints:**

- `fp_downcast_rounding` can only be set during floating-point downcast, otherwise an error will be reported
- When `bitcast=True`, no numeric conversion is performed; rounding/overflow modes are ignored
- `overflow_mode` is only meaningful for integer types (Ascend extension)

### 2.2 DataType Support Table

| Support | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|---------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|:-------:|:----:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### 2.3 Shape Support Table

Supports any number of dimensions and any shape size.

### 2.4 Special Limitations

None

### 2.5 Usage

**Basic Usage:**

```python
import triton
import triton.language as tl

@triton.jit
def cast_example():
    # Create float32 tensor
    x = tl.zeros([2, 3], dtype=tl.float32)

    # Convert to int32
    y = tl.cast(x, tl.int32)

    return y

## Call example
result = cast_example()
print(result.dtype)  # Output: int32
```

**Advanced Usage:**

```python
@triton.jit
def cast_advanced_example():
    # Create float32 tensor
    x = tl.zeros([2, 3], dtype=tl.float32)

    # Bit-level reinterpretation
    y = x.cast(tl.int32, bitcast=True)

    # Floating-point downcast, round toward zero
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")

    # float32 → int8, enable saturation mode (Ascend extension, values exceeding int8 range are clamped to [-128, 127])
    w = x.cast(tl.int8, overflow_mode="saturate")

    return y, z, w
```

**Practical Application Scenario:**

```python
@triton.jit
def quantization_kernel(x_ptr, output_ptr, scale, zero_point, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Load float32 data
    x = tl.load(x_ptr + offsets, mask=mask)

    # Quantization: convert to int8
    x_quantized = tl.cast(x * scale + zero_point, tl.int8, overflow_mode="saturate")

    # Store quantized result
    tl.store(output_ptr + offsets, x_quantized, mask=mask)
```