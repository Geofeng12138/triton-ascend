# triton.language.randn

## 1. OP Overview

Description: Given 1 seed scalar and 1 offset block, returns 1 float32 random block from **N**(**0**,**1**) (standard normal distribution).
Prototype:

```python
triton.language.randn(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter Name | Type                | Description                                                             |
| -------------- | ------------------- | ----------------------------------------------------------------------- |
| `seed`         | `int` or `tensor`   | Seed used for generating random numbers                                 |
| `offset`       | `int` or `tensor`   | Offset used for generating random numbers                               |
| `n_rounds`     | `constexpr`, default 10 | Number of iterations for the Philox algorithm                        |

Return Value:
1 float32 random block, with the same shape as `offset`, whose values follow the standard normal distribution `N(0, 1)`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

Input seed type:

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |

#### 2.2.2 Shape Support

No special requirements

### 2.3 Special Limitations

> Missing community capability and cannot be implemented

### 2.4 Usage Example

The following example demonstrates the call to `randn`:

```python
import math
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    offsets = block_offset + tl.arange(0, XBLOCK)  # Block-level offset tensor
    mask = offsets < N
    rand_vals = tl.randn(5, 10 + offsets, n_rounds)  # Generate a block of random numbers at once
    tl.store(x_ptr + offsets, rand_vals, mask=mask)

shape = (1024,)
y_calf = torch.zeros(shape, dtype=torch.float32).npu()
numel = y_calf.numel()
ncore = 1 if numel < 32 else 32
xblock = math.ceil(numel / ncore)
kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)
```