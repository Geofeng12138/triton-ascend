# triton.language.multibuffer

## 1. OP Overview

Introduction: Sets multi-buffering for a tensor, allowing the compiler to create multiple copies of the same tensor.
Prototype:

```python
triton.language.multibuffer(
    src,
    size,
    _builder=None
) -> None
```

## 2. OP Specification

### 2.1 Parameter Description

| Parameter Name | Type                | Description                                                      |
| -------------- | ------------------- | ---------------------------------------------------------------- |
| `src`          | `tensor`            | The source tensor to be multi-buffered                           |
| `size`         | `int` or `constexpr`| The number of buffer copies to create                            |
| `_builder`     | -                   | Reserved parameter, not supported for external calls             |

Return Value:
`None`: This operation is a compilation hint, it does not return a value at runtime and only affects the compiler's optimization behavior.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|            | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ---------- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    |

#### 2.2.2 Shape Support

Supports tensors of any shape.

### 2.3 Special Restrictions

| Restricted Parameter | Description                                                                      |
| -------------------- | -------------------------------------------------------------------------------- |
| `size`               | The current implementation only supports `size` equal to `2`.                    |

### 2.4 Usage Example

The following example demonstrates how to set multi-buffering for the tensor `tmp0` in a kernel, combined with other compilation hints:

```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        # Set double buffering for tmp0
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```