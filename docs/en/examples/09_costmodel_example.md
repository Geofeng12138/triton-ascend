# Costmodel End-to-End Example

This example demonstrates the basic invocation flow of the costmodel backend:

- Generate TTIR using Triton frontend operators;
- Construct `costmodel_bench` inputs for multiple candidate configs;
- Call `costmodel_bench` to obtain the predicted latency for each config.

This flow is suitable for quickly filtering out configs with poor expected performance before autotuning. The example only uses a vector addition kernel to focus on the costmodel's inputs and return values.

## Complete Example

Save the following code as `costmodel_example.py` and run it:

```python
from __future__ import annotations

import triton
import triton.language as tl
from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import make_backend
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def make_ttir(kernel, signature, constants):
    source = ASTSource(kernel, signature, constants, attrs=None)
    target = GPUTarget("npu", "", 32)
    backend = make_backend(target)

    options = backend.parse_options(
        {
            "num_warps": 8,
            "num_stages": 2,
            "debug": False,
            "multibuffer": False,
            "compile_mode": "simd",
            "enable_costmodel_backend": True,
            **source.parse_options(),
        }
    )

    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return str(ast_to_ttir(kernel, source, context, options, {}, {}))


signature = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
}
n_elements = 98432
configs = [
    {"name": "block256", "BLOCK_SIZE": 256},
    {"name": "block1024", "BLOCK_SIZE": 1024},
    {"name": "block2048", "BLOCK_SIZE": 2048},
]

items = []
for cfg in configs:
    ttir = make_ttir(add_kernel, signature, {"BLOCK_SIZE": cfg["BLOCK_SIZE"]})
    items.append(
        {
            "config": cfg["name"],
            "ttir": ttir,
            # n_elements is the 4th parameter in the signature, corresponding to %arg3 in TTIR.
            # pid_x provides a static estimate for tl.program_id(0).
            "arg_bindings": f"arg3={n_elements},pid_x=0",
        }
    )

latencies = costmodel_bench(items)
for config, latency_us in sorted(latencies.items(), key=lambda item: item[1]):
    print(f"{config}: {latency_us:.3f} us")
```

## Example Output

Different versions of costmodel parameters may cause slight variations in the specific values, but the output structure is similar:

```text
block256: 0.098 us
block1024: 0.110 us
block2048: 0.126 us
```

The return value of `costmodel_bench` is a dictionary, where the key is the input `config` and the value is the predicted latency in microseconds. The upper-level autotune logic can sort by value and prioritize retaining configs with faster predictions.

## Key Points

1. `ASTSource + ast_to_ttir` only generates TTIR and does not actually compile or launch the kernel.
2. `config` affects `tl.constexpr`, such as `BLOCK_SIZE`, so each candidate config needs its own TTIR generated.
3. Each element received by `costmodel_bench` must contain at least `config` and `ttir`, and can optionally include `arg_bindings`.
4. `arg_bindings` is used to bind runtime integer parameters to `%argN` in TTIR. For example, in this case, `n_elements=98432` corresponds to `arg3=98432`.
5. If the kernel uses `tl.program_id(0)`, `pid_x=0` is typically required. If `tl.num_programs(0)` is also used, you can additionally pass `num_programs_x=...`.