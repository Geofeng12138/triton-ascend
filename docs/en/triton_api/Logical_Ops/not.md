# triton.language.tensor.__not__

## 1. Function Overview

Description: Performs element-wise logical NOT on a tensor (0 becomes 1, non-zero becomes 0). Corresponds to Python's `not` keyword — Triton handles this specially through an AST visitor, rewriting `not X` as `X.__not__()`. This differs from bitwise NOT `~X` (see [invert](./invert.md)): the former is logical NOT, while the latter is bitwise inversion.

```python
# Via the not keyword (intercepted by Triton AST)
not x

# Or by directly calling the dunder method
x.__not__()
```

## 2. Specifications

### 2.1 Parameter Description

| Parameter Name | Type       | Description                                                    |
| -------------- | ---------- | -------------------------------------------------------------- |
| `x`            | `tensor`   | Tensor data                                                    |
| `_semantic`    | -          | Reserved parameter, external invocation not supported currently |

Return value:
`out`: The shape of the output tensor is the same as the shape of the input `x`.

### 2.2 OP Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×      | ×      | ×      | ×     | ×    | ×    | ×    | ×    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | ×    | ×    | ×    | ×    | √    |

Conclusion: Ascend additionally supports non-bool types compared to GPU.

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | ------------------------- |
| GPU    | Only supports 1~5D tensors |
| Ascend A2/A3 | Only supports 1~5D tensors |

Conclusion: In terms of shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.

### 2.3 Special Limitations

> Missing capabilities relative to the community that cannot be implemented

None currently.

### 2.4 Usage Example

The following example performs element-wise bitwise NOT on the input tensor `x`:

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr, z_ptr,
            XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr,
            XNUMEL: tl.constexpr, YNUMEL: tl.constexpr, ZNUMEL: tl.constexpr):
    xoffs = tl.program_id(0) * XB
    yoffs = tl.program_id(1) * YB
    zoffs = tl.program_id(2) * ZB

    xidx = tl.arange(0, XB) + xoffs
    yidx = tl.arange(0, YB) + yoffs
    zidx = tl.arange(0, ZB) + zoffs

    idx = xidx[:, None, None] * YNUMEL * ZNUMEL + yidx[None, :, None] * ZNUMEL + zidx[None, None, :]

    X = tl.load(x_ptr + idx)
    Y = tl.load(y_ptr + idx)

    ret = not(X)

    tl.store(output_ptr + idx, ret)

x = test_common.generate_tensor(shape, dtype).npu()
```