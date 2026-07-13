# triton.language.static_assert

## 1. Function Overview

`static_assert` is used to assert whether a condition holds at compile time. If the condition is not met, compilation fails. This is a compile-time checking tool that does not require setting debug environment variables.

```python
triton.language.static_assert(cond, msg='', _semantic=None)
```

## 2. Specification

### 2.1 Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cond` | `bool` | Required | The condition expression to assert at compile time |
| `msg` | `str` | `''` | Error message displayed when assertion fails |
| `_semantic` | - | - | Reserved parameter, external calls not supported |

### 2.2 Type Support

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

**Note:** The type of the value in the `cond` statement must be `constexpr`.

### 2.3 Usage

```python
import triton.language as tl

@triton.jit
def basic_static_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # Basic assertion: check if BLOCK_SIZE is a power of 2
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    # Assertion with custom error message
    tl.static_assert(BLOCK_SIZE >= 64, "BLOCK_SIZE must be at least 64 for performance")

    # Using non-constant values in static_assert conditions will cause compilation errors
    # val = tl.load(x_ptr)
    # tl.static_assert(val <= 64)
```