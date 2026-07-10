# triton.language.tensor.logical_or

## 1. OP Overview

Description: Performs element-wise logical OR operation on two tensors.

```python
x.logical_or(y)
```

Called as a member function of `tensor`, e.g., `x0.logical_or(x1)`.

## 2. OP Specification

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `input` | `tensor` | Tensor data, left operand, representing the primary data for comparison |
| `other`   | `tensor` | Tensor data, right operand, performs element-wise logical OR with `input` |
| `_builder` | - | Reserved parameter, external calls not supported |

Return value:
`tl.tensor`: A tensor with the same shape as `input`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | × | × | × | × | × | × | × | × | × | × | √ |
| Ascend A2/A3 | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | × | √ | √ |

Conclusion: In terms of DataType, Ascend additionally supports integer and floating-point types (except fp64, fp8) compared to GPU.

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| -------- | ---------------------- |
| GPU    | No restrictions |
| Ascend A2/A3 | No restrictions |

Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.

### 2.3 Special Limitations

> Relative community capability missing and cannot be implemented

None.

### 2.4 Usage

The following example demonstrates performing a logical OR operation on three-dimensional tensors `x0` and `x1`:

```python
@triton.jit
def triton_logical_or_3d(in_ptr0, in_ptr1, out_ptr0, XB, YB, ZB, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L) + tl.program_id(0) * XB
    mblk_idx = tl.arange(0, M) + tl.program_id(1) * YB
    nblk_idx = tl.arange(0, N) + tl.program_id(2) * ZB
    idx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    x1 = tl.load(in_ptr1 + idx)
    ret = x0.logical_or(x1)
    odx = lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    tl.store(out_ptr0 + odx, ret)
```