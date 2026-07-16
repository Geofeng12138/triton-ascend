# triton.heuristics

```python
triton.heuristics(values)
```

A decorator used to specify how to compute certain metaparameter values. This is useful in cases where auto-tuning is too expensive or not applicable.

```python
@triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
@triton.jit
def kernel(x_ptr, x_size, **META):
    BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size
```

**Parameters:** `values (dict[str, Callable[[list[Any]], Any]]**)` - A dictionary containing metaparameter names and functions that compute the metaparameter values. Each such function accepts a list of positional arguments as input.