# Quick Start

## Project Introduction
Triton-Ascend is an optimized version of Triton adapted for Huawei Ascend NPUs, designed for efficient kernel function auto-tuning, operator compilation, and deployment. By maintaining compatibility with Triton's core syntax and performing deep optimizations for Ascend NPU features, it helps users quickly develop and deploy high-performance computing tasks on the Ascend platform.
This guide uses a vector addition example deployed via a software package on Ubuntu 22.04 to help users get started with Triton-Ascend quickly.

## Software Package Installation

### Environment Preparation

#### Hardware Requirements

Supported operating systems: linux(aarch64/x86_64)

Supported Ascend products: Atlas A2/A3/A5 series

Minimum hardware configuration: Single card with 32GB memory (recommended)

#### Software Dependencies

Determine and install the Python, CANN, and torch_npu software versions. This step is required for both software package installation and source code compilation.
-   Python version selection: py3.9-py3.11 are all supported.

-   CANN version selection: Visit the Ascend community official website and follow the provided <a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">community software installation guide</a> to complete the CANN installation and configuration. It is recommended to download and install version 9.0.0.

-   torch_npu version selection: The current compatible torch_npu version is 2.7.1.post4.

### Specific Implementation (Using whl Package Installation as an Example)
```bash
# Taking the installation of triton-ascend 3.2.1 as an example
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```
Note: For triton-ascend 3.2.1 and above, Triton-Ascend mitigates installation overwrite issues by declaring Triton as an installation dependency. When installing Triton-Ascend, the community Triton is installed first, and then Triton-Ascend overwrites the directory with the same name, thus preventing subsequent installations of other software packages that depend on Triton from reinstalling Triton and overwriting Triton-Ascend.

## Quick Start

### Example 1: Run the Vector Addition Example from tutorials to Verify Results

Vector addition example: [01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
By comparing the output results of the Triton kernel function with the native PyTorch computation, it demonstrates that the Ascend NPU device can correctly invoke the Triton kernel function and ensure computational accuracy.

```bash
# Set CANN environment variables (using the default installation path `/usr/local/Ascend` for the root user as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Clone the triton-ascend source repository and examples (optional, required when running examples without source code compilation installation)
git clone https://github.com/triton-lang/triton-ascend.git
# Run the tutorials example:
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

Observing similar output indicates that the environment is configured correctly.

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

### Example 2: From GPU to NPU: Migrating Triton

Triton-Ascend maintains full syntax compatibility with the community Triton. By simply replacing the **tensor device declaration** and a few `torch.cuda.*` interfaces, existing GPU examples can run on the Ascend NPU.
This section provides a simple vector addition test example, using the most basic migration method with minimal modifications to the original GPU code, demonstrating the complete migration process to help users quickly experience the workflow of migrating GPU scripts to the Ascend NPU.

The GPU version example file `test_add.py` is as follows:

```python
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@pytest.mark.parametrize('SIZE,BLOCK_SIZE', [(98432, 1024)])
def test_add(SIZE, BLOCK_SIZE):
    device_id = torch.cuda.current_device()
    device = torch.device('cuda', device_id)

    x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    y = torch.randn(SIZE, device='cuda', dtype=torch.float32)

    output_cpu = torch.empty(SIZE, dtype=torch.float32)
    output = output_cpu.cuda()

    def grid(meta):
        return (triton.cdiv(SIZE, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    torch.cuda.synchronize()

    output_torch = x + y
    assert_close(output, output_torch, rtol=1e-3, atol=1e-3)
```

Migration only requires replacing the GPU-related APIs with the corresponding NPU versions. The mapping is as follows:

| GPU Syntax                      | NPU Syntax                      |
| ------------------------------- | ------------------------------- |
| `device='cuda'`                 | `device='npu'`                  |
| `tensor.cuda()`                 | `tensor.npu()`                  |
| `torch.cuda.current_device()`   | `torch.npu.current_device()`    |
| `torch.cuda.synchronize()`      | `torch.npu.synchronize()`       |

Kernel functions annotated with `@triton.jit` use the Triton general language and generally do not require special modification. The Launch grid calling convention is also identical to GPU.

The core changes are shown in diff format:

```diff
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

# ...（kernel code remains unchanged）...

@pytest.mark.parametrize('SIZE,BLOCK_SIZE', [(98432, 1024)])
def test_add(SIZE, BLOCK_SIZE):
-   device_id = torch.cuda.current_device()
+   device_id = torch.npu.current_device()

-   x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
-   y = torch.randn(SIZE, device='cuda', dtype=torch.float32)
+   x = torch.randn(SIZE, device='npu', dtype=torch.float32)
+   y = torch.randn(SIZE, device='npu', dtype=torch.float32)

    output_cpu = torch.empty(SIZE, dtype=torch.float32)
-   output = output_cpu.cuda()
+   output = output_cpu.npu()

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

-   torch.cuda.synchronize()
+   torch.npu.synchronize()

    output_torch = x + y
    assert_close(output, output_torch, rtol=1e-3, atol=1e-3)
```
After modification, you can run the test case using `pytest`. Successful execution indicates a successful migration.
```bash
pytest test_add.py
```
If the `pytest` component is not installed, you can install it using `pip install pytest`.