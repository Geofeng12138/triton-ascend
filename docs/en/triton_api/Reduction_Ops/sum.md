# triton.language.sum

## 1. OP Overview

Description: `triton.language.sum` computes the sum of elements of the input tensor along the specified axis and returns the result.

```python
triton.language.sum(input, axis=None, keep_dims=False)
```

> **Version Difference Note**
>
> The `dtype` parameter is a feature introduced in community Triton 3.5.0. The currently released Triton-Ascend is developed based on community Triton 3.2.0 and does not include the `dtype` parameter. When upgrading to community Triton 3.5.0 in the future, full support for the `dtype` parameter will be provided.

## 2. OP Specification

### 2.1 Parameter Description

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | `Tensor` | Input tensor |
| `axis` | `int` or `None` | The dimension along which to perform the sum operation. If `None`, sum over all dimensions |
| `keep_dims` | `bool` | If `True`, keep the reduced dimensions with length 1 |

Return value:
`tensor`: The result of summing the elements of the input tensor along the specified axis.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU Support | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

#### 2.2.2 Shape Support

Conclusion: There is no difference in Shape support between GPU and Ascend platforms.

### 2.3 Special Limitations

> Relative to community capabilities that are missing and cannot be implemented
> `keep_dims=True` requires more test cases to determine full support. Currently, it has been tested and supports `keep_dims=True` for 3D with dim=2.
> The `dtype` parameter is not yet supported in the current version. In community Triton 3.5.0, the `dtype` parameter controls the accumulation data type for the sum operation: when not specified, integer types with bit width less than 32 are automatically promoted to `int32`/`uint32` to avoid overflow; when explicitly specified, the input is first converted to the specified type before performing the sum. The current Triton-Ascend is based on community Triton 3.2.0, and this type promotion logic is not yet supported. Full support will be provided when upgrading to version 3.5.0.

### 2.4 Usage Example

The following example implements a sum operation on a 2D tensor:

```python
@triton.jit
def tt_sum_2d(in_ptr, out_ptr,
              xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
              XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr, dim: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    idx = xidx[:, None] * ynumel + yidx[None, :]

    x = tl.load(in_ptr + idx)
    ret = tl.sum(x, dim)

    if dim == 0:
        oidx = yidx
    else:
        oidx = xidx
    tl.store(out_ptr + oidx, ret)

```