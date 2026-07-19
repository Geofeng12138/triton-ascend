# triton.language.atomic_xchg

## 1. OP Overview

Description: Atomic exchange operation, performs an atomic exchange at the specified memory location.
Prototype:

```python
triton.language.atomic_xchg(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```

Can be called as a member function of a tensor, e.g., `x.atomic_xchg(...)`, which is equivalent to `atomic_xchg(x, ...)`.

## 2. OP Specification

### 2.1 Parameter Description

| Parameter Name | Type                | Description                                                             |
| -------------- | ------------------- | ----------------------------------------------------------------------- |
| `pointer`      | `triton.PointerDType` | The memory location to operate on. Updates *pointer to val, and writes the computed result back to this memory location. |
| `val`          | `pointer.dtype.element_ty` | The target value to update to (right operand).                          |
| `mask`         | `int1` or `tensor<int1>`, optional | Specifies the data range to prevent out-of-bounds access.               |
| `sem`          | `str`, optional     | Specifies the memory semantics of the operation.<br>Community official accepted values are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, previous release operations are visible (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, can be read).<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous write operations). |
| `scope`        | `str`, optional     | The thread scope for observing the synchronization effect of the atomic operation.<br>Accepted values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (representing "SYSTEM").<br>We only support "gpu". |
| `_semantic`    | -                   | Reserved parameter, temporarily not supported for external calls.        |

Return value:
`pointer`: tensor, the old value before the operation was performed.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU      | ×      | ×    |  √      | ×    |  ×    | √   | √    | √      | ×    | √      | √    | ×    |  ×    |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √     | √     | √      | √    | √    | ×    | ×    | ×    |

#### 2.2.2 Shape Support

No special requirements, default eight-dimensional.

### 2.3 Special Limitation Notes

> Relative to community capabilities, these are missing and cannot be implemented.

| Difference Point | Description                                                                           |
| ---------------- | ------------------------------------------------------------------------------------- |
| `sem`            | Community official accepted values are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel". |
| `scope`          | Accepted values are "gpu", "cta", or "sys".<br>We only support "gpu".                  |

### 2.4 Usage Example

The following example implements atomic exchange computation:

```python
@triton.jit
def atomic_xchg(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_xchg(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0), tmp1, xmask)
```