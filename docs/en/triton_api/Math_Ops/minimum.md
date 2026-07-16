# triton.language.minimum

## 1. Function Overview

Description: Computes the element-wise minimum of x and y.

```python
triton.language.minimum(x, y, propagate_nan: ~triton.language.core.constexpr = <PROPAGATE_NAN.NONE: 0>, _semantic=None)¶
```

## 2. Specification

### 2.1 Parameter Description

| Parameter Name | Type               | Description                                                    |
| -------------- | ------------------ | -------------------------------------------------------------- |
| `x`            | `tensor`           | Tensor data                                                    |
| `y`            | `tensor`           | Tensor data                                                    |
| `propagate_nan`| `tl.PropagateNan`  | Whether to propagate NaN values                                |
| `_semantic`    | -                  | Reserved parameter, not supported for external calls currently |

Return value:
`x`: A tensor with the same shape as the input x

### 2.2 OP Specification

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | √     | ×      | ×      | ×      | √     | √    | √    | ×    | √    | √    |

Conclusion: Compared to GPU, Ascend lacks fp64 support.

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| ------ | ------------------------- |
| GPU    | Only supports 1~5D tensors |
| Ascend A2/A3 | Only supports 1~5D tensors |

Conclusion: In terms of shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.

### 2.3 Special Limitations

> Missing capabilities relative to the community that cannot be implemented

None.

#### 2.3.1 propagate_nan Parameter Limitation

**Note: When `propagate_nan=tl.PropagateNAN.NONE`, the system automatically adds NaN value handling logic, which leads to:**

1. **Increased UB space usage**: Additional NaN detection and handling consumes more UB space
2. **Potential performance degradation**: Due to the added computation logic, operator execution performance may decrease

**Recommendations:**

- If the input data does not contain NaN values, or strict NaN handling semantics are not required, it is recommended to use the default value or select an appropriate `propagate_nan` parameter value based on actual needs
- In scenarios with tight UB space, special attention should be paid to the selection of this parameter to avoid compilation failures due to insufficient UB space

### 2.4 Usage Example

The following example implements the element-wise minimum of input tensors `x` and `y`:

```python
@triton.jit
def fn_npu_(output_ptr, x_ptr, y_ptr,
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

    ret = tl.minimum(X, Y)

    tl.store(output_ptr + idx, ret)

```