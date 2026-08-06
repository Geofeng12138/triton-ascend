# Installation Guide

**Triton-Ascend** is an optimized version of Triton adapted for Huawei Ascend processors, mainly used to provide efficient kernel auto-tuning, operator compilation, and deployment capabilities. It supports Ascend Atlas A2/A3/950 series products. While being compatible with Triton's core syntax, it has been deeply optimized for Ascend NPU characteristics, including automatic parsing of kernel parameters, optimization of memory access logic, and improvement of secure deployment mechanisms.

## Environment Preparation

**Hardware Requirements**

- Ascend products: Support Atlas A2/A3/950 series.

- NPU configuration: At least single-card 32GB memory is recommended.

- Operating system: Linux system is required. For details, please refer to the <a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">Compatibility Query Assistant</a>. All subsequent operations in this article are demonstrated in the **Ubuntu** environment.

**Software Dependencies**

Determine the CANN, Python, and TorchNPU software versions and install them. For this, you can refer to the "[CANN Quick Installation](https://www.hiascend.com/cann/download)" guide on the Ascend community official website to complete the driver and firmware installation.

- CANN version: 9.0.0
- Python version: python3.11
- TorchNPU version: 2.7.1.post4

## Quick Installation

```bash
pip install triton-ascend --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

<a id="install-from-source"></a>

## Source Installation

### Install Dependencies

```bash
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11 # build-time dependencies
```

### Build Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
git checkout main
pip install -e .
```

### Custom LLVM Build (Optional)

If you need to customize the LLVM build process, you can follow the steps below to compile Triton-Ascend.

1. **Code Preparation**: Check out the LLVM source code of a specified version using `git checkout` and apply the patch.

    ```bash
    git clone --no-checkout https://github.com/llvm/llvm-project.git
    cd llvm-project
    git checkout f6ded0be897e2878612dd903f7e8bb85448269e5
    wget https://raw.githubusercontent.com/triton-lang/triton-ascend/refs/heads/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch
    git apply llvm_patch_f6ded0b.patch
    ```

2. **Build LLVM**: The path `/path/llvm-install` is the LLVM installation path planned by the user, which needs to be adjusted according to the actual situation; the path `{PATH_TO}` is the path where the user checked out the LLVM source code in step 1.

    ```bash
    export LLVM_INSTALL_PREFIX=/path/llvm-install
    cd {PATH_TO}/llvm-project
    mkdir build
    cd build
    cmake ../llvm \
        -G Ninja \
        -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
        -DCMAKE_LINKER=/usr/bin/lld-15 \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
    ninja install

    cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
    ```

3. **Compile Triton-Ascend**

    ```bash
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME="triton-ascend" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    python3 setup.py install
    ```

## Development Image

### Check Image Version

**Table 2** CANN version and image tag mapping table.
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">CANN Version</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Chip Type</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Python Version</th>
    <th style="width: 40%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Image Tag</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-910b-ubuntu22.04-py3.10</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-a3-ubuntu22.04-py3.10</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">8.5.0-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.11</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-950-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A2</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.12</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-910b-ubuntu22.04-py3.12</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.12</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-a3-ubuntu22.04-py3.12</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.12</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">9.0.0-950-ubuntu22.04-py3.12</td>
  </tr>
</table>

### Using the Image

```bash
# Here we take 9.0.0-a3-ubuntu22.04-py3.11 as an example
docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container \
--security-opt seccomp=unconfined \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /home:/home \
-v /etc/ascend_install.info:/etc/ascend_install.info \
quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11 \
/bin/bash

# Enter the container, and install Triton-Ascend using either the quick installation or the source installation described above
docker exec -u root -it triton-ascend_container /bin/bash
```

## Running Examples

**Run the vector addition example in tutorials to verify the result**

Vector addition example: <a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>

```bash
# Set CANN environment variables (taking the root user default installation path `/usr/local/Ascend` as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Pull the triton-ascend source repository and examples (no need to pull again if Triton-Ascend was installed from source)
git clone https://github.com/triton-lang/triton-ascend.git
# Run the tutorials example
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

Observing similar output indicates that the environment is configured correctly:

```text
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## Installation FAQ

**Question 1: The error "ERROR: No matching distribution found for torch==2.7.1+cpu" occurs when installing TorchNPU**

**Solution**

You can try manually installing torch before installing TorchNPU:

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

**Question 2: When compiling and installing Triton-Ascend, if GCC < 9.4.0, the error "ld.lld: error: unable to find library -lstdc++fs" may be reported**

**Solution**

This error is generally caused by the linker being unable to find the stdc++fs library. This library is used to support file system features of versions earlier than GCC 9. In this case, you need to manually uncomment the following related code snippet in the CMake file.
File path: triton-ascend/CMakeLists.txt

```bash
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```

**Question 3: When running an operator, the error "ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package" is reported**

**Root Cause Analysis**

The triton-ascend directory is overwritten by triton, causing triton-ascend functionality to be damaged.

**Solution**

Uninstall the damaged triton-ascend and reinstall it. Taking version 3.2.1 as an example, you can run the following command to fix it:

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

**Question 4: Why does Triton-Ascend 3.2.1 add a dependency on triton?**

Answer: Triton-Ascend is a secondary development based on Triton and shares the same installation directory name with Triton. If users install Triton-Ascend and then install triton or third-party packages that depend on triton, the triton directory will be overwritten, causing Triton-Ascend functionality to be damaged.
Therefore, by adding the triton dependency, the following reminder will appear when triton is overwritten and installed.

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton-ascend 3.2.1 requires triton==3.5.0, but you have triton 3.5.1 which is incompatible.
```

If users encounter this and want to restore Triton-Ascend functionality, they can do the following:

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

```

**Question 5: Why are the Triton versions that Triton-Ascend 3.2.1 depends on inconsistent?**

Answer: X86 and Arm use different versions of the community Triton installation package because the community has been providing the X86 installation package since Triton 3.2, while the Arm installation package has been provided since Triton 3.5.

**Question 6: How to Confirm Chip Type**

You can use the npu-smi command to check the NPU model on your system. For example, in the output of the npu-smi info command, "910B4" corresponds to chip type A2 (Ascend 910b series):

```Text
root@localhost:/# npu-smi  info
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                                    |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B4               | OK            | 82.6                 32                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                2871 / 32768            |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| No running processes found in NPU 0
