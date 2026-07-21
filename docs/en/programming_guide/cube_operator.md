# Cube Operator Development

Cube operators primarily perform matrix multiplication or batched matrix multiplication as their main computational workload, with `tl.dot` as the core in Triton code. The key to Cube operators is designing tiles around the M/N/K dimensions, enabling efficient movement of A/B tiles to on-chip memory and accumulation on the Cube Core.

## Simple Cube Operator Development

For simple Cube operators, refer to the [matrix multiplication example](../examples/05_matrix_multiplication_example.md) in this repository. A minimal development path includes:

1. Define input and output shapes and strides, e.g., `A[M, K]`, `B[K, N]`, `C[M, N]`.
2. Use `tl.program_id` to map the current program to the `(pid_m, pid_n)` tile of the output matrix.
3. Construct 2D offsets for A and B using `BLOCK_SIZE_M/N/K`.
4. Loop along the K dimension to load A/B sub-blocks and accumulate into an fp32 accumulator using `tl.dot`.
5. Convert the accumulator to the output dtype and write back to C with boundary masks.

The core structure is as follows:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  stride_am: tl.constexpr, stride_ak: tl.constexpr,
                  stride_bk: tl.constexpr, stride_bn: tl.constexpr,
                  stride_cm: tl.constexpr, stride_cn: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

When tuning simple Cube operators, prioritize:

- Whether `BLOCK_M/N/K` meets hardware support and UB/L1 capacity constraints.
- Whether the K-dimension loop can enable `multibuffer` to pipeline data movement and computation.
- Whether the output tile includes additional bias, scale, or activation. If post-processing is lightweight, it can still be classified as a Cube operator; if post-processing involves significant Vector reduction or cross-core synchronization, it should be organized as a CV fusion operator.

## Complex Cube Operator Development

Complex Cube scenarios typically arise from attention, batched matmul, grouped matmul, or irregularly shaped matrix multiplications. The complex cases in the main branch of [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops) are concentrated in `tutorial/best_practice/`, where [`002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py) serves as a reference for complex Cube core logic: it includes two `tl.dot` operations (QK and PV) and demonstrates how to reorganize K/V memory access under discrete KV cache indexing.

For complex Cube operators, it is recommended to decompose in the following order:

1. **First extract the pure matrix multiplication core**: Confirm the input tile shape, dtype, accumulation dtype, and output tile shape for each `tl.dot`.
2. **Then handle irregular memory access**: If the K/V cache has low-dimensional discreteness and high-dimensional continuity, direct 2D loading may degrade to scalar access. First load into UB along the continuous dimension, then reorganize into the layout required by `tl.dot` via transpose or the Ascend extension interface `extension.insert_slice`.
3. **Leave reduction and normalization to clearly defined boundaries**: For example, `max/sum/exp` in attention are Vector logic. If placed in the same kernel as `tl.dot`, switch to the [CV fusion operator development](./cv_fusion_operator.md) approach.
4. **Design inner loops for long K or long sequences**: The K-dimension loop should control the on-chip occupancy of single A/B tiles; the sequence-dimension loop should avoid loading overly large K/V blocks at once.
5. **Use Autotune to manage candidate tiles**: Prepare multiple sets of `BLOCK_M/N/K` and `multibuffer` configurations for common shapes, allowing the runtime to select the optimal combination.

A common risk in complex Cube operators is directly migrating a large number of programs from GPU to NPU. If the number of output tiles far exceeds the number of physical Cube Cores, consider having each program process multiple tiles via an inner loop, or set `TRITON_ALL_BLOCKS_PARALLEL=1` to reduce scheduling overhead when logical cores are confirmed to be independent.