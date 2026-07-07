# triton.language.make_tensor_descriptor

## 1. OP Overview

Description: Creates a tensor descriptor object.
Prototype (Triton 3.4.0):

```python
triton.language.make_tensor_descriptor(
    base: tensor,
    shape: List[tensor],
    strides: List[tensor],
    block_shape: List[constexpr],
    _semantic=None
) -> tensor_descriptor
```

## 2. OP Specification

### 2.1 Parameter Description

| Parameter    | Type                | Description                                                                                     |
| ------------ | ------------------- | ----------------------------------------------------------------------------------------------- |
| `base`       | `tensor`            | Base pointer of the tensor                                                                      |
| `shape`      | `List[tensor]`      | Shape of the tensor                                                                             |
| `strides`    | `List[tensor]`      | List of strides for each dimension of the tensor, with the following constraints: - Leading dimensions must be multiples of 16 bytes - The last dimension must be contiguous |
| `block_shape`| `List[constexpr]`   | Shape of the block to be loaded/stored from global memory                                       |
| `_semantic`  | -                   | Reserved parameter, external calls are not supported                                            |

Return value:
`tensor_descriptor`: Tensor descriptor object (cannot be directly used for arithmetic operations; must be used with `load` / `store`)

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

|       | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
| ----- | ----- | ---- | ------ | ----- | ------ | ----- | ------ | ----- | ---- | ---- | ---- | --------- |
| GPU   | √     | √    | √      | √     | √      | √     | √      | √     | √    | √    | √    | ×         |
| Ascend A2/A3 | √     | √    | ×      | √     | ×      | √     | ×      | √     | √    | √    | √    | ×         |

#### 2.2.2 Shape Support

|              | Supported Dimension Range |
| ------------ | ------------------------- |
| GPU          | Only supports 1~5D tensors |
| Ascend A2/A3 | Only supports 1~5D tensors |

Conclusion: In terms of Shape, there is no difference between GPU and Ascend platforms; both support 1 to 5-dimensional tensors.

### 2.3 Special Limitations

> Community capability gaps that cannot be implemented

Conclusion: Compared to GPU, Ascend lacks support for uint16, uint32, and uint64 (hardware limitation).

| Difference Point             | Description                                                                                                                         | Solution                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Binding usage restriction    | `make_tensor_descriptor` / `load_tensor_descriptor` / `store_tensor_descriptor` must be used together, cannot be mixed with `tl.load()` / `tl.store()`. | Upgrading to Triton 3.4.0 to synchronize upstream functions (e.g., `cast`) can resolve this. |
| `padding_option` parameter not supported | The current community main branch has added the `padding_option` parameter for out-of-bounds element padding strategies.            | Can be supported via software development.                                                   |
| Triton version compatibility | Triton 3.2.0 has compatibility issues with some functions (e.g., `cast`). It is recommended to upgrade to Triton 3.4.0 to fix binding restrictions. | Upgrade to Triton 3.4.0.                                                                     |

### 2.4 Usage Example

The following example implements an in-place absolute value calculation on the input tensor `x`:

```python
@triton.jit
def inplace_abs(in_out_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    # Create tensor descriptor
    desc = tl.make_tensor_descriptor(
        in_out_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )
    # Calculate the offset for the current thread
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    # Load data, compute absolute value, store result
    value = desc.load([moffset, noffset])
    desc.store([moffset, noffset], tl.abs(value))

## Initialize tensor
M, N = 256, 256
x = torch.randn(M, N, device="npu")
## Configure block size and grid
M_BLOCK, N_BLOCK = 32, 32
grid = (M // M_BLOCK, N // N_BLOCK)
inplace_abs[grid](x, M, N, M_BLOCK, N_BLOCK)
```