# triton.jit

```python
triton.jit(fn: T) -> JITFunction[T]
triton.jit(*, version=None, repr: Callable | None = None, launch_metadata: Callable | None = None, do_not_specialize: Iterable[int] | None = None, debug: bool | None = None, noinline: bool | None = None) -> Callable[[T], JITFunction[T]]
```

A decorator for JIT-compiling functions using the Triton compiler.

- Note: When calling a JIT-compiled function, if an argument has a `.data_ptr()` method and a `.dtype` attribute, it will be implicitly converted to a pointer.

- Note: This function will be compiled and run on the GPU. It can only access:
    - Python primitives,
    - Built-in functions within the Triton package,
    - Arguments of this function,
    - Other JIT-compiled functions.

**Parameters:** `fn (Callable)` - The function to be JIT-compiled