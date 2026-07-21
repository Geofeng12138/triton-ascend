# Triton-Ascend FAQ

## 1. Installation and Environment Configuration

**Q: How to correctly install Triton-Ascend? Does it support direct pip installation?**

A: You can install it directly using pip

```Python
pip install triton-ascend
```

**Q: Can community Triton and Triton-Ascend coexist?**

A: For triton-ascend 3.2.0 and below, no. You need to uninstall the community Triton first, then install Triton-Ascend.<br>
For triton-ascend 3.2.1 and above, Triton-Ascend mitigates the installation overwrite issue by declaring Triton as an installation dependency.
When installing Triton-Ascend, the community Triton is installed first, and then Triton-Ascend overwrites the directory with the same name, thus preventing subsequent installations of other packages that depend on Triton from overwriting Triton-Ascend.
The reason x86 and arm use different versions of the community Triton installation package is that the community only started providing arm version packages from version 3.5 onwards: x86 depends on triton==3.2.0, arm depends on triton==3.5.0.

- Note: If, after installing triton-ascend, you install a third-party package that depends on triton, or triton itself, it will overwrite the installed Triton-Ascend directory.
In this case, you need to uninstall the community Triton and Triton-Ascend first, then reinstall Triton-Ascend.

```Python
pip uninstall triton
pip uninstall triton-ascend
pip install triton-ascend
```

**Q: Can Triton-Ascend be used on non-Ascend hardware (e.g., CUDA AMD)?**

A: No, Triton-Ascend can only be used on Ascend NPU hardware environments.

## 2. Accuracy and Numerical Consistency Issues

**Q: The NPU execution result is inconsistent with the PyTorch/CPU/GPU reference result. How to debug?**

A: For examples, please refer to [07_accuracy_comparison_example.md](../zh/examples/07_accuracy_comparison_example.md)
For debugging methods, please refer to [Interpreter Mode Debugging Method](./debug_guide/debugging.md#5-调试方法)

## 3. Error Codes and Exception Handling

**Q: Why does the kernel compilation report MLIRCompilationError? How to locate the specific failing Pass?**

A: Please refer to [Compilation Error Debugging Method](./debug_guide/debugging.md#52-编译错误调试方法)

## 4. Debugging and Logging

**Q: How to enable detailed log output? Where is TRITON_DEBUG=1 output?**

A: You can use TRITON_DEBUG=1 to obtain detailed debug dump files. Please refer to [Debug Dump Files](./debug_guide/debugging.md#32-调试转储文件dump-files)

**Q: Can I print intermediate tensor values inside a kernel? Is tl.device_print available?**

A: You can use tl.device_print to print tensors inside a kernel. Please refer to [Print Debugging Method](debug_guide/debugging.md#51-打印调试方法)

## 5. Development and Contribution

**Q: How to build and test Triton-Ascend locally?**

A: For local build and test methods, please refer to [Installing Triton-Ascend from Source](./installation_guide.md#源码编译安装)

**Q: What CI checks does a PR need to pass?**

A: The CI checks for a PR include: code security and style checks, open-source snippet checks, malicious code checks, compilation build, and developer tests.

## 6. Performance Tuning

**Q: Is there a performance analysis tool (profiler) available?**

A: Yes, there is an integrated performance analysis tool (profiler). Please refer to [Operator Performance Tuning Method](./debug_guide/profiling.md)

## 7. UB Overflow Common Issues

**Q: The compilation reports a "UB Overflow" error. How to resolve it?**

A: UB Overflow is a common issue in Triton-Ascend development. Please refer to the [UB Overflow Troubleshooting Guide](./debug_guide/ub_overflow.md) to diagnose the problem. If you don't know how to reduce tiling to decrease UB usage, you can use Autotune to automatically select the optimal configuration. For using Autotune, please refer to [Triton-Ascend Autotune Usage Guide](./autotune_guide.md).
Operators that run on the A5 may cause UB Overflow when migrated to A2/A3 due to differences in UB size. If manual troubleshooting fails, Autotune can also be used to automatically select the optimal configuration.

## 8. Triton Usage Limitations

**Q: What are the usage limitations for pointer parameters in Triton Kernels?**

A: Triton-Ascend assumes at compile time that all externally input pointer parameters essentially point to different memory regions and cannot recognize Pointer Alias scenarios. When multiple pointer parameters actually point to the same memory at runtime, but the compiler cannot know this fact, it may lead to optimization failures or abnormal execution results. For example:

```Python
@triton.jit
def func(ptr0, ptr1):
    # load from ptr0 and do something
    # store to ptr0
    # load from ptr1 and do something
    # store to ptr1

in_out_tensor = torch.randn(shape)
func[grid](in_out_tensor, in_out_tensor)
```

In the above code, `ptr0` and `ptr1` actually point to the same memory (i.e., the same `in_out_tensor`), but the compiler cannot recognize this pointer alias relationship. Therefore, writing code where the same tensor is passed as multiple pointer parameters is not supported, and the corresponding Kernel will not be able to enable related optimizations.

**Q: What are the limitations of using `tl.load` / `tl.store` within control flow operations like `if` / `for` / `while`?**

A: Triton-Ascend supports memory access using a pointer from the same source after simple address updates within control flow. Placing `tl.load` / `tl.store` inside control flow is also a reasonable practice.
However, it is not recommended to merge pointers from different sources or with different structures after control flow and then perform unified memory access. It is also not recommended to repeatedly update pointer states and simultaneously perform store/read-after-write operations within complex nested control flow.

The current version has incomplete support for scenarios combining `if` / `for` / `while` with `tl.load` / `tl.store`. Subsequent versions will continue to improve this. Currently, it is recommended to follow the limitations below.

It is not recommended to merge pointers with different base addresses, or block pointers constructed in different branches, after the branches and then perform memory access:

```Python
if cond:
    ptr = x + offsets
else:
    ptr = y + offsets
value = tl.load(ptr)
```

It is recommended to place the memory access within each respective branch, merging the loaded values (rather than pointers or block pointers) after the branches:

```Python
if cond:
    value = tl.load(x + offsets)
else:
    value = tl.load(y + offsets)
```