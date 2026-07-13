# triton.language.zeros

## 1. OP Overview

Description: `triton.language.zeros` returns a tensor filled with the scalar value 0 of the given shape and dtype.

```python
triton.language.zeros(shape, dtype)
```

## 2. OP Specification

### 2.1 Parameter Description

| Parameter Name | Type                  | Description                                   |
| ------------- | ----------------- | ---------------------------- |
| `shape`           | `tuple of ints`               | Shape of the new array, e.g., (8, 16) or (8, ) |
| `dtype`            | `tl.dtype`               |  Data type of the new array, e.g., tl.float16  |

Return Value:
`tensor`: Returns a tensor filled with the scalar value 0 of the given shape and dtype.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | × |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | × |

#### 2.2.2 Shape Support

Conclusion: There is no difference between GPU and Ascend platforms regarding Shape.

### 2.3 Special Limitations

> Relative community capability missing and cannot be implemented
> None

### 2.4 Usage Example

The following example demonstrates returning a tensor of shape (XB, YB, ZB) filled with the value 0:

```python
@triton.jit
def fn_f32(output_ptr, x_ptr, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    xidx = tl.arange(0, XB)
    yidx = tl.arange(0, YB)
    zidx = tl.arange(0, ZB)

    idx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    X = tl.load(x_ptr + idx)

    ret = tl.zeros((XB, YB, ZB), dtype=tl.float32)

    oidx = xidx[:, None, None] * YB * ZB + yidx[None, :, None] * ZB + zidx[None, None, :]

    tl.store(output_ptr + oidx, ret)
```