# triton

| API | Brief Description |
|-----|------------------|
| [jit](./jit.md) | JIT decorator - Compiles functions using the Triton compiler |
| [autotune](./autotune.md) | Decorator for auto-tuning a function compiled with `triton.jit` |
| [heuristics](./heuristics.md) | Decorator for specifying how to compute certain meta-parameter values |
| [Config](./Config.md) | An object representing possible kernel configurations that the auto-tuner can try |

```{toctree}
:maxdepth: 3
:hidden:

jit.md
autotune.md
heuristics.md
Config.md
```