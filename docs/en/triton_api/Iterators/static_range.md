# triton.language.static_range

## 1. Function Overview

`static_range` is a static range iterator, similar to `range` but performs aggressive loop unrolling optimization at compile time.

```python
triton.language.static_range(arg1, arg2=None, step=None, _semantic=None)
```

## 2. Specification

### 2.1 Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arg1` | `constexpr` | Required | Start value (when used as a single argument, it serves as the end value, starting from 0) |
| `arg2` | `constexpr` | - | End value (exclusive) |
| `step` | `constexpr` | `1` | Step increment for each iteration |
| `_semantic` | - | - | Reserved parameter, external calls not supported |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend 910 Series | ✓ | ✓ | ✓ | ×|×| × | × | ✓ | × | × | × | × | × |

### 2.3 Special Limitations

> Relative community capability missing and cannot be implemented

Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).

### 2.4 Usage

```python
@triton.jit
def optimized_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    # Use static_range for small-scale loop unrolling to eliminate loop overhead
    for i in tl.static_range(BLOCK_SIZE):
        # When BLOCK_SIZE is a compile-time constant, the entire loop is unrolled
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)

    # Comparison: using range incurs loop control overhead
    for i in tl.range(BLOCK_SIZE):
        # This loop has loop control logic at runtime
        x = tl.load(x_ptr + i)
        y = x * x
        tl.store(y_ptr + i, y)
```

`static_range` trades code size for runtime performance, suitable for scenarios with known and small loop counts.