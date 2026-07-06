# triton.language.rand

## 1. OP Overview

Description: Given 1 seed scalar and 1 offset block, returns 1 random block of float32 type in **U**(**0**,**1**).
Prototype:

```python
triton.language.rand(
 seed,
 offset,
 n_rounds: constexpr = 10
)
```

## 2. OP Specifications

### 2.1 Parameter Description

| Parameter    | Type               | Description                                                    |
| ------------ | ------------------ | -------------------------------------------------------------- |
| `seed`       | `int` or `tensor`  | Seed used to generate random numbers                           |
| `offset`     | `int` or `tensor`  | Offset used to generate random numbers                         |
| `n_rounds`   | `constexpr`, default 10 | Number of iteration rounds for the Philox algorithm            |

Return Value:
1 random block of float32 type, with the same shape as the offset, uniformly distributed in the interval `[0.0, 1.0)`

### 2.2 Supported Specifications

#### 2.2.1 DataType Support

Input seed type:

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √    | √     | √     |√     | ×    | ×    | ×    | ×    | √    |

#### 2.2.2 Shape Support

No special requirements

### 2.3 Special Limitations

> Not supported compared to community capabilities

### 2.4 Usage Example

The following example demonstrates a call to rand:

```python
@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds) # Generate a random number for each index
        tl.store(x_ptr + global_offset, rand_vals) # Store the random number

y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()
numel = y_calf.numel()
ncore = 1 if numel < 32 else 32
xblock = math.ceil(numel / ncore)
kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)
```