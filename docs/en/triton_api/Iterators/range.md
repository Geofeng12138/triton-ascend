# triton.language.range

## 1. Function Overview

`range` is an iterator that counts upward, similar to Python's `range()` function, but allows more parameters to be passed.

```python
triton.language.range(arg1, arg2=None, step=None, num_stages=None, loop_unroll_factor=None, disallow_acc_multi_buffer=False, flatten=False, warp_specialize=False, disable_licm=False, _semantic=None)
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arg1` | `int` / `constexpr` | Required | Start value (when single parameter, acts as end value, starting from 0) |
| `arg2` | `int` / `constexpr` | - | End value (exclusive) |
| `step` | `int` / `constexpr` | `1` | An integer, the step increment for each iteration |
| `num_stages` | `int` | - | Number of pipeline stages (number of iterations executed concurrently) |
| `loop_unroll_factor` | `int` | - | Loop unroll factor (< 2 means no unrolling) |
| `disallow_acc_multi_buffer` | `bool` | `False` | Disable multi-buffer optimization for dot operation accumulators |
| `flatten` | `bool` | `False` | Automatically flatten nested loops into a single loop |
| `warp_specialize` | `bool` | `False` | Enable warp specialization (Blackwell GPU only) |
| `disable_licm` | `bool` | `False` | Disable loop-invariant code motion optimization |
| `_semantic` | - | - | Reserved parameter, external calls not supported |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | × | × |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | × | × | ✓ | × | × | × | × | × |

### 2.3 Special Limitation Notes

> Relative community capability missing and unimplementable

Ascend lacks support for uint8, uint16, uint32, uint64, and fp64 compared to GPU (hardware limitation).
The functionalities related to `disallow_acc_multi_buffer`, `flatten`, `warp_specialize`, and `disable_licm` are not yet complete.

### 2.4 Usage

```python
import triton.language as tl

@triton.jit
def basic_examples():
    # Single parameter: 0 to 9
    for i in tl.range(10):
        # i = 0, 1, 2, ..., 9
        pass

    # Two parameters: 2 to 9
    for i in tl.range(2, 10):
        # i = 2, 3, ..., 9
        pass

    # Three parameters: 0 to 10, step 2
    for i in tl.range(0, 10, 2):
        # i = 0, 2, 4, 6, 8
        pass
```

```python
@triton.jit
def advanced_examples():
    # Using loop optimization parameters
    for i in tl.range(0, 100, num_stages=3, loop_unroll_factor=4):
        # Pipeline stages = 3, loop unroll factor = 4
        pass

    # Nested loop flattening
    for i in tl.range(0, 10, flatten=True):
        for j in tl.range(0, 20, flatten=True):
            # Both loops will be automatically flattened into a single loop
            pass
```