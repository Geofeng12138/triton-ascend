# CV Fusion Operator Development

CV fusion operators refer to operators that simultaneously use Cube Core and Vector Core: Cube Core is typically responsible for `tl.dot`, matrix multiplication, or convolution-style main computation, while Vector Core handles bias, activation, softmax, reduction, mask, layout rearrangement, or cross-block synchronization. The goal of CV fusion is to reduce kernel boundaries and GM round trips, but it requires simultaneous control over Cube tile, Vector tile, UB/L1 occupancy, and synchronization relationships.

## Simple CV Fusion Operator Development

For simple CV fusion, it is recommended to first extract a stable `tl.dot` main computation from the [matrix multiplication example](../examples/05_matrix_multiplication_example.md) in this repository, then add Vector post-processing before writing back; for more complex slice updates, refer to the [fused attention example](../examples/04_fused_attention_example.md). The minimal path is as follows:

1. First implement a stable Cube main computation, e.g., `acc = tl.dot(a, b, acc)`.
2. Fuse lightweight Vector post-processing before writing back the accumulator, e.g., bias, scale, activation, or dtype cast.
3. For larger accumulators, use `range` with `extension.extract_slice`/`extension.insert_slice` for ordinary sub-block splitting to avoid UB overflow during the Vector post-processing stage.
4. `extension.parallel(..., bind_sub_block=True)` is a stronger explicit multi-Vector sub-block binding path. It may not be available when target hardware and compilation configurations differ, so it is not recommended as the default writing style for simple examples.

Example structure:

```python
# Inside the matmul kernel, after the K loop completes, a fp32 accumulator is obtained.
acc = tl.dot(a, b, acc)  # Typically located inside the K dimension loop; shown here for structure only.

# Fuse lightweight Vector post-processing before writing back.
acc = tl.where(acc >= 0, acc, 0.01 * acc)
c = acc.to(tl.float16)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tl.store(c_ptrs, c, mask=c_mask)
```

When developing simple CV fusion, keep boundaries clear: Cube is responsible for generating larger 2D accumulators, while Vector handles element-wise operations or small-scale reductions within the same tile. If the Vector part needs to share state across multiple Cube tiles, synchronization, workspace, or kernel splitting needs to be introduced.

## Complex CV Fusion Operator Development

For complex CV fusion, refer to the best practices in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops):

- [`tutorial/best_practice/002-decode_grouped_attention.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.py): In Decode attention, QK/PV uses Cube, while softmax, mask, exponent, normalization, and discrete KV memory access rearrangement use Vector.
- [`tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md): Demonstrates how to use `extension.insert_slice`, transpose, and kernel splitting optimization to reduce discrete memory access and padding overhead when fusing cat, slice, and conv1d update.

It is recommended to organize complex CV fusion by data flow layers:

1. **Main Computation Layer**: Identify which steps must use Cube, e.g., QK, PV, GEMM, batched matmul.
2. **Vector Post-processing Layer**: Identify whether softmax, activation, mask, scale, normalization, cat/slice, layout transform, etc., can be completed within the same tile.
3. **Memory Access Rearrangement Layer**: For discrete KV cache, MoE token rearrangement, and tail-axis tensors, prioritize using `extension.insert_slice`, `extension.extract_slice`, transpose, or axis borrowing transpose in UB to form hardware-friendly contiguous access.
4. **Pipeline and Synchronization Layer**: Explore overlapping execution of Cube and Vector through compilation options such as `multibuffer`, `set_workspace_multibuffer`, `tile_mix_vector_loop`, `tile_mix_cube_loop`.
5. **Kernel Splitting Layer**: CV fusion operators are typically launched with a grid based on the number of Cube Cores; at runtime, Vector Cores collaborate at approximately a 1:2 ratio. Do not simply adopt the large grid approach used on GPUs.

For attention-like CV fusion, it is recommended to first get non-causal, short sequence, small head_dim cases working, then gradually add:

- Causal mask processing in stages.
- Long sequence K/V block loops.
- Numerically stable softmax updates for `m_i`/`l_i`.
- Accumulator workspace and sub-block splitting when HEAD_DIM is large.
- Load rearrangement under discrete indices for KV cache.

When tuning complex CV fusion, prioritize observing the time proportion of Cube, Vector, and MTE2 in profiling. If Cube is waiting for Vector, consider reducing the granularity of Vector post-processing or enabling CV balance related options; if Vector is waiting for data movement, first check discrete memory access, tail-axis padding, and multibuffer configuration.