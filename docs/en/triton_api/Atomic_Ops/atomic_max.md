# triton.language.atomic_max

## 1. OP Overview

Introduction: Atomic maximum operation, performs an atomic max operation at the specified memory location.
Prototype:

```python
triton.language.atomic_max(
    pointer,
    val,
    mask=None,
    sem=None,
    scope=None,
    _semantic=None
) -> pointer
```

Can be called as a member function of a tensor, e.g., `x.atomic_max(...)`, which is equivalent to `atomic_max(x, ...)`.

## 2. OP Specification

### 2.1 Parameter Description

| Parameter Name | Type               | Description                                                        |
| -------------- | ------------------ | ------------------------------------------------------------------ |
| `pointer`      | `triton.PointerDType` | The memory location to operate on. The result of max(*pointer, val) is written back to this memory. |
| `val`          | `pointer.dtype.element_ty` | The value to compare with the target memory.                       |
| `mask`         | `int1` or `tensor<int1>`, optional | Specifies the data range to prevent out-of-bounds access.          |
| `sem`          | `str`, optional    | Specifies the memory semantics of the operation.<br>Accepted values in the upstream community are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel":<br>- acquire: After acquiring the lock, can see previous release operations (equivalent to a "read" operation that blocks until the "latest" data, i.e., data released by other threads, is readable).<br>- release: All operations before releasing the lock are visible to threads that subsequently acquire the lock (equivalent to a "write" operation that "synchronizes" all previous writes). |
| `scope`        | `str`, optional    | The thread scope for observing the synchronization effect of the atomic operation.<br>Accepted values are "gpu" (default), "cta" (cooperative thread array, thread block), or "sys" (representing "SYSTEM").<br>We only support "gpu". |
| `_semantic`    | -                  | Reserved parameter, not supported for external calls currently.    |

Return value:
`pointer`: tensor, the old value before the operation.

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | √     | ×     | ×      | ×      | ×      | √     | ×    | √    | ×    | ×    | ×    |
| Ascend A2/A3 | √ | √     | √     | ×     | ×      | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |

Conclusion: Compared to GPU, Ascend lacks support for int64.

#### 2.2.2 Shape Support

No special requirements.

### 2.3 Special Limitations

> Capabilities missing relative to the upstream community and cannot be implemented.

| Difference Point | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| Data Type        | Compared to GPU, Ascend lacks support for int64 (hardware limitation).           |
| `sem`            | Accepted values in the upstream community are "acquire", "release", "acq_rel" (default, representing "ACQUIRE_RELEASE"), and "relaxed".<br>We only support "acq_rel". |
| `scope`          | Accepted values are "gpu", "cta", or "sys".<br>We only support "gpu".            |

### 2.4 Usage Example

The following example implements an atomic maximum operation:

```python
@triton.jit
def triton_atomic_max(
    in_ptr0, out_ptr0, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_max(out_ptr0 + (x1), tmp0, xmask)
```