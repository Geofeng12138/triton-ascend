# Vector Operator Development

Vector operators are primarily executed by the Vector Core. Typical forms include element-wise computation, row-level reduction, type conversion, Gather/Scatter, Mask updates, and small fusion operators that do not contain `tl.dot`. The development focus is not on making the grid as fine as possible, but rather, given a fixed number of physical Vector Cores, having each program process multiple tiles in a loop within the core.

## Simple Vector Operator Development

For simple Vector operators, you can start with the [vector addition example](../examples/01_vector_add_example.md) in this repository or `third_party/ascend/tutorials/01-vector-add.py`. The basic steps for this type of operator are as follows:

1. Use `tl.arange` to construct the continuous offset of the current tile.
2. Use `mask` to protect the tail block and avoid out-of-bounds load/store.
3. Complete the element-wise computation and write back the result.
4. When the number of grids is much larger than the number of physical cores, fix the grid to `num_vectorcore` and use `range(pid, num_blocks, num_core)` within the kernel to process in batches.

The basic kernel structure is as follows:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_idx in range(pid, num_blocks, num_core):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
```

During development, prioritize checking three types of issues:

- **Data Type**: The Ascend Vector unit has different support and performance for different integer types. For index, length, and offset data that do not affect precision, prefer using `int32`. Refer to `triton-ascend-ops/tutorial/basic/001-vector_add.zh.md` and `002-vector_cmp.zh.md`.
- **BLOCK_SIZE**: BLOCK_SIZE should be as large as possible within the UB capacity. If a UB overflow occurs, first reduce the number of elements processed at a time, then consider splitting into sub-blocks.
- **Number of Cores**: The number of physical Vector Cores on an NPU is typically a few dozen. When migrating GPU code that uses small tiles and a large grid to the NPU, significant overhead can easily arise from multiple rounds of dispatch.

## Complex Vector Operator Development

Complex Vector operators usually do not consist of a single element-wise expression but involve combined logic with discrete memory access, batch rearrangement, multiple outputs, or long hidden sizes. Refer to the following examples in [Ascend/triton-ascend-ops](https://github.com/Ascend/triton-ascend-ops):

- [`tutorial/best_practice/004-gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/004-gather_scatter.py): Ascend-friendly implementation of Megablocks gather/scatter/scatter_wgrad.
- [`tutorial/best_practice/005-binned_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/005-binned_gather_scatter.py): Gather/scatter grouped by expert/bin.
- [`tutorial/best_practice/006-padded_gather_scatter.py`](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/006-padded_gather_scatter.py): MoE gather/scatter with padding.

The organization method for this type of operator is typically:

1. **Partition the outer task by physical cores**: Use `num_vectorcore` as the grid, with each program responsible for a segment of indices or tokens.
2. **Partition the hidden dimension by UB capacity**: Use `BLOCK_X` to chunk `NUM_COLUMNS`, reserving space for double buffers, indices, and temporary tensors.
3. **Use `SUB_BLOCK_SIZE` to merge small-grained discrete tasks**: Load a group of indices at once, organize them into continuous temporary blocks in UB, reducing GM scalar memory access and multiple stores.
4. **Use extended semantics to manage local data within UB**: First, import the extension module via `import triton.language.extra.cann.extension as extension`, then use `extension.insert_slice` to merge multiple rows of data, and use `extension.extract_slice` to extract sub-blocks before writing back in a scattered manner.
5. **Reserve a unified mask for the tail block**: In complex gather/scatter, there are simultaneously index masks, column masks, and expert/bin boundaries. It is recommended to name them separately and only combine them at load/store points.

A typical UB budget approach is as follows:

```python
num_core = get_npu_properties()["num_vectorcore"]
block_size = triton.cdiv(indices_length, num_core)
align_elems = 16
block_x = triton.cdiv(min(num_columns, max_block_x), align_elems) * align_elems
sub_block_size = max((ub_budget - block_x * element_bytes) //
                     (block_x * element_bytes + index_bytes), 1)
```

When the performance of a complex Vector operator does not meet expectations, prioritize investigating the following directions:

- Whether the grid is much larger than the number of physical Vector Cores, leading to multiple rounds of dispatch.
- Whether discrete memory access can be transformed into "batch loading into UB followed by selection within UB".
- Whether the trailing axis satisfies 32B alignment; if not, whether transposition or borrowing an axis for transposition can be used to avoid automatic padding.
- Whether `BLOCK_X` and `SUB_BLOCK_SIZE` cause UB overflow or result in too small a transfer granularity.