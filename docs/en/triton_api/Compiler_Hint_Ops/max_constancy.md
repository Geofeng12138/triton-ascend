# triton.language.max_constancy

## 1. Function Overview

`max_constancy` is used to declare the constancy pattern of values in an input tensor to the compiler, informing the compiler that each group of consecutive values in the input data are equal.

```python
triton.language.max_constancy(input, values, _builder=None, _semantic=None)
```

## 2. Specification

### 2.1 Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `Tensor` | Required | Input tensor whose values have a specific constancy pattern |
| `values` | `constexpr[int]` or `list[constexpr[int]]` | Required | Compile-time constant integer (or sequence of integers) describing the constancy pattern |
| `_semantic` | - | - | Reserved parameter, external calls not supported |

**`values` describes the constancy characteristics of each dimension, so the dimension of `values` must be the same as the dimension of `input`.
Note the dimension reduction case when the last dimension of `shape` is `1`.**

For example: a 2D `input` corresponds to a general `values` parameter of `[1,1]`.

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.3 Special Limitations

> Missing relative to community capabilities and cannot be implemented

Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

### 2.4 Usage

```python
@triton.jit
def basic_constancy_example(A, B, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input_data = tl.load(A + offsets)

    # Use constexpr to declare constancy pattern: every 4 consecutive values are equal
    # Example input pattern: [0,0,0,0,1,1,1,1,2,2,2,2,...]
    input_data = tl.max_constancy(input_data, [4])

    # The compiler can optimize based on this information
    result = input_data * 2
    tl.store(B + offsets, result)
```