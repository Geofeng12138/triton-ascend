# Quick Installation

This document mainly introduces how to quickly complete the installation of the **Triton-Ascend** basic suite in an Ubuntu environment. For detailed operation steps, please refer to [<u>Installation Guide</u>](#installation-guide).

## Quick Setup Based on Docker Image

Directly use the out-of-the-box image released by Triton-Ascend to quickly build a development environment.

### Confirm the Image
**Table 1** Partial mapping table of Ascend chips, corresponding products, and image tags. For more images, refer to the [OVERVIEW.zh.md](../../docker/OVERVIEW.zh.md) document.
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Chip Model</th>
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Corresponding Product</th>
    <th style="width: 34%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Image Tag</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Ascend 910b</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 800T A2, Atlas 900 A2 PoD</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-910b-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Ascend A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 800T A3</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-a3-ubuntu22.04-py3.11</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Ascend 950</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Atlas 950PR Series</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">3.2.1-950-ubuntu22.04-py3.11</td>
  </tr>
</table>

### Specific Implementation
1.  Create a container

    ```bash
    # Assuming your NPU device model is A3, the device is installed on /dev/davinci1, and your NPU driver is installed at /usr/local/Ascend:
    # Using image_tag: 3.2.1-a3-ubuntu22.04-py3.11 as an example:
    container_name=triton-ascend_container
    image_tag=3.2.1-a3-ubuntu22.04-py3.11
    docker run -u 0 -dit --shm-size=512g --name=${container_name} --net=host --privileged \
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
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home:/home \
    quay.io/ascend/triton:${image_tag} \
    /bin/bash
    ```

2.  Enter the container
    ```bash
    docker exec -it triton-ascend_container bash
    ```
3. Clone the code
```bash
# Clone the triton-ascend source repository and examples
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
```
4. Run the example: <a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>
```bash
# Run the tutorials example:
python3 ./third_party/ascend/tutorials/01-vector-add.py
```
Observing similar output indicates the environment is configured correctly:
```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

# Installation Guide <a id = "installation-guide" ></a>

## Overview

Triton-Ascend is an optimized version of Triton adapted for Huawei Ascend chips. It primarily provides efficient kernel function auto-tuning, operator compilation, and deployment capabilities. It supports Ascend Atlas A2/A3 series products and is compatible with core Triton syntax while being deeply optimized for Ascend NPU characteristics, including automatic parsing of kernel function parameters, optimizing memory access logic, and improving secure deployment mechanisms.

This document mainly introduces three installation methods for Triton-Ascend: Package Installation; Image Installation; Source Code Compilation and Installation.

## Hardware and Operating System

-   Ascend Products: Supports Atlas A2/A3/A5 series.

-   NPU Configuration: At least 32GB memory per single card is recommended.

-   Operating System: Requires a Linux system. For specific choices, please refer to the <a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">Compatibility Query Assistant</a>. All subsequent operations in this document are demonstrated using an Ubuntu environment.

## Installation Method Selection

Quick Decision: Most users can directly choose package installation; choose image installation for containerized deployment; choose source code compilation for secondary development or code modification.
**Table 2** Comparison of different installation methods
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 24.41%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Installation Method</th>
    <th style="width: 19.15%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Target Audience</th>
    <th style="width: 26.21%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Core Advantages</th>
    <th style="width: 30.23%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Reason for Choice</th>
  </tr>
    <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Package Installation</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Production environment users, operations personnel</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Simple installation, automatic dependency management, easy upgrade/uninstall</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Seeking stability, quick deployment, avoiding environment setup hassle</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Source Code Compilation</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Developers, users needing custom features</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">High customizability, supports latest features</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Need to modify source code, adapt to special hardware or features</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Image Installation</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Quick experience users, containerized deployment needs</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">One-click startup, environment isolation, no manual dependency configuration</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">Want to run the process as fast as possible, or need multi-environment consistency</td>
  </tr>
</table>

### Package Installation

#### Related Product Version Compatibility

<table style="table-layout: fixed; width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
    <thead>
    <tr>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Triton-Ascend Version</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Python Supported Versions</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>CANN Version</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Torch-NPU Version</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Notes</strong>
    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
   <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.1</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x<br>Python3.12.x<br>Python3.13.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">9.0.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.7.1.post4<br>2.8.0.post4<br>2.9.0.post2<br>2.10.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x does not support aarch64</td>
    </tr>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">8.5.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.6.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">NA</td>
    </tr>
    <tr>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.0rc4</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">8.5.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.6.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">NA</td>
    </tr>
    </tbody>
</table>

#### Check the Installation Environment<a id = "environment-preparation" ></a>
Determine and install the CANN, Python, and Torch-NPU software versions. This step is required for both package installation and source code compilation.
-   Recommended CANN Version: 9.0.0
-   Recommended Python Version: python3.11
-   Recommended Pytorch Version: 2.7.1
-   Recommended Torch-NPU Version: 2.7.1.post4.

#### whl Package Installation
1.  Check Python Version

    ```bash
    python3 --version
    ```
    If the command output is as follows, it means the Python version is 3.11.15:
    ```text
    root@test:/# python3 --version
    Python 3.11.15
    ```

2.  Install the whl Package
    -    For Triton-Ascend 3.2.0 and below, Triton-Ascend and Triton cannot coexist. You need to uninstall the community Triton first, then install Triton-Ascend.
    -    For Triton-Ascend 3.2.1 and above, Triton-Ascend mitigates the installation overwrite issue by declaring Triton as an installation dependency. See [FAQ](#appendix-faq) for details.

    ```bash
    # Example for installing triton-ascend 3.2.1
    pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
    ```


### Source Code Compilation and Installation
If you need to develop or customize **Triton-Ascend**, you can use the source code compilation and installation method. After preparing the installation environment and dependencies, it is recommended to use the [<u>Online Installation</u>](#quick-installation) method to complete the source-based installation; if there are special requirements, such as the target machine not being able to connect to the internet, you can perform [<u>Offline Installation</u>](#manual-installation).


#### Check the Installation Environment
Determine and install the CANN, Python, and torch_npu software versions. This step is required for both package installation and source code compilation. Refer specifically to the [<u>Environment Preparation</u>](#environment-preparation) section of the Package Installation.


**System Recommendations**
**Table 3** Recommended PyTorch Compatibility Versions
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Pytorch Version</th>
    <th style="width: 33%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Recommended GCC Version</th>
    <th style="width: 34%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Recommended GLIBC Version</th>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.7.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">11.2.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.8.0</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">13.3.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
  <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.9.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">13.3.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
    <tr style="height: 50px;">
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">PyTorch2.10</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">13.3.1</td>
    <td style="border: 1px solid #ddd; padding: 8px; text-align: left;">2.28</td>
  </tr>
</table>

#### Install Dependencies
1.  Install System Library Dependencies
    Install zlib1g-dev / lld / clang. Optionally install the ccache package to speed up builds.
    -   Recommended clang version >= 15
    -   Recommended lld version >= 15
    ```bash
    apt update
    apt install zlib1g-dev clang-15 lld-15
    apt install ccache # optional
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
    ```
2.  Install Python Dependencies
    ```bash
    pip install ninja cmake wheel pybind11 # build-time dependencies
    ```


#### Online Installation<a id = "quick-installation" ></a>
```bash
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend
git checkout main

# Optional. If you have a pre-built LLVM locally, you can specify the local LLVM path directly to avoid downloading the LLVM pre-built package. If not, ignore this and proceed with the installation command below.
export LLVM_SYSPATH=/path/to/LLVM

# Execute the installation command
pip install -e .
```

#### Offline Installation - Based on LLVM Build<a id = "manual-installation" ></a>
Triton uses LLVM 22 to generate code for GPU and CPU. Similarly, Ascend's Bisheng compiler also relies on LLVM to generate NPU code, so compiling the LLVM source code is necessary. Please pay attention to the specific version of LLVM required.

##### Code Preparation
Checkout the specific version of the LLVM source code using `git checkout` and apply the patch:
```bash
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout f6ded0be897e2878612dd903f7e8bb85448269e5
wget https://raw.githubusercontent.com/triton-lang/triton-ascend/refs/heads/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch
git apply llvm_patch_f6ded0b.patch
```

##### Build and Install LLVM
-   Step 1: Set the environment variable LLVM_INSTALL_PREFIX to your target installation path
    ```bash
    # The path is the planned LLVM installation path, adjust as needed
    export LLVM_INSTALL_PREFIX=/path/to/llvm-install
    ```
-   Step 2: Execute the following commands to build and install LLVM
    ```bash
    cd {PATH_TO}/llvm_project # The path where you cloned the LLVM code, adjust as needed
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
    ```
-   Step 3: Copy FILECHECK to the target installation path
    ```bash
    cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
    ```


##### Build Triton-Ascend
-   Step 1: Clone Triton-Ascend
    ```bash
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    ```
-   Step 2: Compile and Install Triton-Ascend
    ```bash
    # Ensure the LLVM installation target path ${LLVM_INSTALL_PREFIX} from the [Build LLVM] section is set
    # Ensure clang>=15, lld>=15, ccache are installed

    LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
    TRITON_BUILD_WITH_CCACHE=true \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME="triton-ascend" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    python3 setup.py install
    ```

### Image Installation
Install the Docker environment image using the Dockerfile. Use the pre-built quay.io/ascend/cann image as the base image to skip the CANN installation step, significantly speeding up the build process.

#### Check Image Versions

**Table 4** CANN Version and Image Tag Mapping
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


#### Image Installation
1.  Build the Image

    ```bash
    # Using 9.0.0-a3-ubuntu22.04-py3.11 as an example
    git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend
    docker build \
    --build-arg CANN_BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11 \
    -t triton-ascend-image:latest -f ./docker/Dockerfile .
    ```

2.  Start the Container
    ```bash
    docker run -u 0 -dit --shm-size=512g --name=triton-ascend_container --net=host --privileged \
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
    triton-ascend-image:latest \
    /bin/bash

    # Enter the container
    docker exec -u root -it triton-ascend_container /bin/bash
    ```


## Installation Verification
Install runtime dependencies:
```bash
# Clone the triton-ascend source repository and examples (optional, needed only if running examples without source compilation)
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend && pip install -r requirements.txt
```

Run the example: <a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>
```bash
# Set CANN environment variables (using the default installation path `/usr/local/Ascend` for root user as an example)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# Run the tutorials example:
python3 ./third_party/ascend/tutorials/01-vector-add.py
```
Observing similar output indicates the environment is configured correctly:
```
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

# Appendix: FAQ<a id = "appendix-faq" ></a>

## Error "ERROR: No matching distribution found for torch==2.7.1+cpu" when installing torch_npu

### Solution
Try manually installing torch first, then install torch_npu:
```
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

## Error "ld.lld: error: unable to find library -lstdc++fs" when compiling Triton-Ascend if GCC < 9.4.0

### Solution
This error is usually caused by the linker being unable to find the stdc++fs library. This library supports filesystem features for versions before GCC 9. You need to manually uncomment the following relevant code snippet in the CMake file.
File path: triton-ascend/CMakeLists.txt
```
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```
## ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package when executing an operator
### Root Cause Analysis
The triton-ascend directory was overwritten by triton, causing damage to the triton-ascend functionality.
### Solution
Uninstall the damaged triton-ascend and reinstall it. Taking version 3.2.1 as an example, execute the following command to fix it:
```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

## Why does Triton-Ascend 3.2.1 add triton as a dependency?
Answer: Triton-Ascend is a secondary development based on Triton and shares the same installation directory name as Triton. If a user installs triton or a third-party package that depends on triton after installing Triton-Ascend, it will overwrite the triton directory, damaging the Triton-ascend functionality.
Therefore, by adding the triton dependency, the following warning will be displayed when triton is overwritten.
```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton-ascend 3.2.1 requires triton==3.5.0, but you have triton 3.5.1 which is incompatible.
```
If users encounter this and want to restore triton-ascend functionality, they can do the following:
```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

```

## Why are the Triton versions depended on by Triton-Ascend 3.2.1 inconsistent?
Answer: The reason x86 and arm use different versions of the community Triton installation package is that the community only started providing arm version installation packages from version 3.5 onwards: x86 depends on triton==3.2.0, arm depends on triton==3.5.0.

## How to confirm the chip type
You can use the npu-smi command to check the NPU model on the system. For example, in the output of the npu-smi info command, "910B4" corresponds to chip type A2 (Ascend 910b series):

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
| No running processes found in NPU 0                                                                              |
```