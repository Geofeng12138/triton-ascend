# ne

## 1. OP Overview

Description: Used to compare elements of two tensors, equivalent to `!=`.

Used as a built-in operator of `tensor`, e.g., `x!=y`.

## 2. OP Specification

### 2.1 Parameter Description

| Parameter | Type | Description |
| :---: | :---: | :---: |
| `input` | `tensor` | Tensor data, left operand, representing the main data to be compared |
| `other`   | `tensor` | Tensor data, right operand, compared element-wise with `input` |
| `_builder` | - | Reserved parameter, external calls not supported |

Return value:
`tl.tensor`: A tensor with the same shape as `input`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ | √ |
| Ascend A2/A3 | √ | √ | √ | × | × | × | × | √ | √ | √ | × | √ | √ |

Conclusion: Compared to GPU, Triton-Ascend lacks support for uint8/uint16/uint32/uint64 and fp64.

#### 2.2.2 Shape Support

|        | Supported Dimension Range |
| -------- | --------------------------- |
| GPU    | No restrictions |
| Ascend A2/A3 | No restrictions |

Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms.

### 2.3 Special Limitations

> Relative to community capabilities, missing and unimplemented

Compared to GPU, Triton-Ascend lacks support for fp64. Support for uint8/uint16/uint32/uint64 types is under development.

### 2.4 Usage Example

The following example implements the `!=` operation on tensors `x0` and `x1`:

```python
@triton.jit
def triton_ne(in_ptr0, in_ptr1, out_ptr0, N: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
       offset = tl.program_id(0) * XBLOCK
       base1 = tl.arange(0, XBLOCK_SUB)
       loops1: tl.constexpr = XBLOCK // XBLOCK_SUB
       for loop1 in range(loops1):
           x_index = offset + (loop1 * XBLOCK_SUB) + base1
           tmp0 = tl.load(in_ptr0 + x_index, mask=x_index < N)
           tmp1 = tl.load(in_ptr1 + x_index, mask=x_index < N)
           tmp2 = tmp0 != tmp1
           tl.store(out_ptr0 + x_index, tmp2, mask=x_index < N)
```