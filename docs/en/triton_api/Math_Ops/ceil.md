# triton.language.ceil

## 1. OP Overview

Description: Computes the ceiling value of each element in a tensor.

```python
triton.language.ceil(x, _semantic=None)
```

## 2. OP Specification

### 2.1 Parameter Description

| Parameter | Type   | Description |
| :---: | :---: | :---: |
| `x` | `tensor` | Tensor data |
| `_semantic`   | - | Reserved parameter, not supported for external calls |

Return value:
`out`: A tensor with the same shape as `x`.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | × | × | × | × | × | × | × | √ | √ | × | × |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| :---: | :---: |
| GPU    | No restrictions |
| Ascend | No restrictions |

Conclusion: In terms of shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Missing relative community capabilities and cannot be implemented

Compared to GPU, Ascend lacks support for fp64 but adds support for fp16 and bf16, and supports integer input.

### 2.4 Usage Example

The following example demonstrates performing the ceiling operation on the input tensor `x`:

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr,
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

    ret = tl.ceil(X)

    tl.store(output_ptr + idx, ret)
```