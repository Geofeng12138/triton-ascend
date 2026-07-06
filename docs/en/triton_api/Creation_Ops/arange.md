# triton.language.arange

## 1. OP Overview

Description: The `triton.language.arange` function generates a contiguous integer sequence from `start` to `end` (excluding `end`).

```python
triton.language.arange(start, end, _semantic=None)
```

## 2. OP Specification

### 2.1 Parameter Description

| Parameter | Type    | Description                                                  |
| --------- | ------- | ------------------------------------------------------------ |
| `start`   | `scalar` | The starting value for creating the contiguous integer sequence. Must be a compile-time constant (`tl.constexpr`). |
| `end`     | `scalar` | The ending value for creating the contiguous integer sequence. |

Return value:
`tensor`: A tensor containing the contiguous integer sequence.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

Conclusion: The parameters `start` and `end` of `arange` must be constants, hence they have no type. The supported value range corresponds to the type, up to `int32`. Hardware instructions also only support up to `int32`.

|                | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| -------------- | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| GPU            | ×     | ×    | ×      | ×     | ×      | √     | ×      | ×     | ×    | ×    | ×    | ×         |
| Ascend A2/A3   | ×     | ×    | √      | ×     | ×      | ×     | ×      | ×     | ×    | ×    | ×    | ×         |

#### 2.2.2 Shape Support

0 =< (end - start) < 1048576
end >= 0, start >= 0

Conclusion: In terms of Shape, there is no difference between the GPU and Ascend platforms.

### 2.3 Special Limitations

> Relative community capability missing and unimplementable

1. The function generates a contiguous integer sequence [start, end). CUDA requires that `range = (end - start)` must be a power of 2. Triton-ascend does not have this requirement.
2. Both NV and Triton-ascend limit the maximum value of `end` to `TRITON_MAX_TENSOR_NUMEL = 1048576`.
3. The input to `arange` must be a constant, supporting values of `uint` and `int` types less than 1048576 (the maximum value `TRITON_MAX_TENSOR_NUMEL`). `int64` is not supported.
4. The `start` and `end` of `arange` must be greater than or equal to 0.

### 2.4 Usage Example

The following example generates a contiguous integer sequence [0, 128):

```python
@triton.jit
def triton_arange(z, BLOCK: tl.constexpr, START: tl.constexpr, END: tl.constexpr):
    off = tl.arange(0, BLOCK)
    val = tl.arange(START, END)
    tl.store(z + off, val)

@pytest.mark.parametrize('param_list',[[0, 128],])
def test_case_access(param_list):
    start, end = param_list
    shape = [end]
    block = end - start
    dtype = 'int32'
    y_cal = torch.zeros(shape, dtype=torch.int32).npu()
    triton_arange[(1, )](y_cal, START = start, END = end, BLOCK = block)
```