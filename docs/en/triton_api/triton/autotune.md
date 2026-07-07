# triton.autotune

```python
triton.autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None, warmup=25, rep=100, use_cuda_graph=False)
```

Decorator for automatic tuning of `triton.jit` functions.

```python
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime the value of x_size changes
)
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE']
```

- Note: When all configs are evaluated, the kernel will be run multiple times. This means any values updated by the kernel will be updated multiple times. To avoid this undesired behavior, use the `reset_to_zero` parameter, which resets the provided tensor values to zero before running any config.
- Note: If the environment variable `TRITON_PRINT_AUTOTUNING` is set to `"1"`, Triton will print a message to stdout after each autotuned kernel, including the time spent on autotuning and the best config.

**Parameters:**

- `configs (list[triton.Config])` - A list of `triton.Config` objects.
- `key (list[str])` - A list of parameter names whose value changes will trigger the evaluation of all configs.
- `prune_configs_by (dict)` - A dictionary of functions for pruning configs. Contains the following fields:
  - `'perf_model'`: A performance model used to predict the runtime of different configs, returning the runtime.
  - `'top_k'`: The number of configs to benchmark.
  - `'early_config_prune'` (optional): A function for early pruning of configs (e.g., `num_stages`). It takes `configs: List[Config]` as input and returns the pruned configs.
- `reset_to_zero (list[str])` - A list of parameter names that will be reset to zero before any config is evaluated.
- `restore_value (list[str])` - A list of parameter names whose values will be restored after evaluating any config.
- `pre_hook (lambda args, reset_only)` - A function to be called before the kernel is invoked. This parameter overrides the default `pre_hook` for `reset_to_zero` and `restore_value`.
  - `args`: The list of arguments passed to the kernel.
  - `reset_only`: A boolean indicating whether the `pre_hook` is only used to reset values, without a corresponding `post_hook`.
- `post_hook (lambda args, exception)` - A function to be called after the kernel is invoked. This parameter overrides the default `post_hook` for `restore_value`.
  - `args`: The list of arguments passed to the kernel.
  - `exception`: The exception raised by the kernel in case of a compilation or runtime error.
- `warmup (int)` - The warmup time passed to the benchmark (in milliseconds), default is 25.
- `rep (int)` - The repetition time passed to the benchmark (in milliseconds), default is 100.
- `use_cuda_graph (bool)` - Whether to use CUDA Graph for performance measurement (default is `False`).