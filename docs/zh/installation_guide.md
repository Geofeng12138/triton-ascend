# 安装指南

**Triton-Ascend** 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，主要用于提供高效的核函数自动调优、算子编译及部署能力，支持 Ascend Atlas A2/A3 等系列产品，兼容 Triton 核心语法的同时，针对昇腾 NPU 特性进行了深度优化，包括自动解析核函数参数、优化内存访问逻辑、完善安全部署机制等。

本文主要介绍 **Triton-Ascend** 的三种安装方式：快速安装；源码编译安装；镜像安装。

## 环境准备

### 硬件要求

- Ascend 产品：支持 Atlas A2/A3/A5 系列。 

- NPU 配置：建议至少单卡 32GB 内存。

- 操作系统：需 Linux 系统，具体选择请参考<a href="https://www.hiascend.com/hardware/compatibility" style="text-decoration: none; color: #0066cc;">兼容性查询助手</a>。本文接下来所有操作均以 Ubuntu 环境演示。

### 软件依赖

确定 CANN、Python、Pytorch 和 Torch-NPU 软件版本并安装。其中，CANN 可以访问昇腾社区官网，根据其提供的<a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">社区软件安装指引</a>完成 CANN 的安装与配置。

- CANN 版本推荐：9.0.0
- Python 版本推荐：python3.11
- Pytorch 版本推荐：2.7.1
- Torch-NPU 版本推荐：2.7.1.post4

**表1** 相关产品版本配套说明表
<table style="table-layout: fixed; width: 100%; border-collapse: collapse; font-family: Arial, sans-serif;">
    <thead>
    <tr>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Triton-Ascend 版本</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Python支持版本</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>CANN 版本</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>Torch-NPU 版本</strong>
    </th>
    <th style="width: 20%; text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd; background-color: #f5f5f5;">
    <strong>备注</strong>
    </th>
    </tr>
    </thead>
    <tbody>
    <tr>
   <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">3.2.1</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x<br>Python3.10.x<br>Python3.11.x<br>Python3.12.x<br>Python3.13.x</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">9.0.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">2.7.1.post4<br>2.8.0.post4<br>2.9.0.post2<br>2.10.0</td>
    <td style="text-align: left; vertical-align: middle; padding: 12px; border: 1px solid #ddd;">Python3.9.x 不支持 aarch64</td>
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

## 安装方式

快速决策：快速安装在此采用的是软件包安装方式，绝大多数用户直接选择 [快速安装](#快速安装) 即可；需要二次开发改代码选 [源码编译安装](#源码编译安装)；需要容器化部署选 [镜像安装](#镜像安装)。

### 快速安装

```bash
# 以安装 triton-ascend 3.2.1 为例
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

### 源码编译安装

#### 安装依赖

```bash
apt update
apt install zlib1g-dev clang-15 lld-15
apt install ccache # optional
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100
update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100
pip install ninja cmake wheel pybind11 # build-time dependencies
```

#### 构建安装 LLVM

```bash
# 检出指定版本的 LLVM 源码并应用补丁
git clone --no-checkout https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout fad3272286528b8a491085183434c5ad4b59ab92
wget https://raw.githubusercontent.com/triton-lang/triton-ascend/6765b03c81c4e9ecb277e4ef1dde61dea0d044f0/third_party/ascend/llvm_patch/fad3272.patch
git apply fad3272.patch

# 路径为用户规划的llvm安装路径,需根据实际调整
export LLVM_INSTALL_PREFIX=/path/to/llvm-install

# 构建和安装 LLVM
cd {PATH_TO}/llvm_project # 路径为用户拉取LLVM代码的路径,需根据实际调整
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

# 拷贝 FILECHECK 到目标安装路径
cp  {PATH_TO}/llvm_project/build/bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck
```

#### 编译 Triton-Ascend

```bash
git clone https://github.com/triton-lang/triton-ascend.git && cd triton-ascend

# 确认已设置 [基于LLVM构建] 章节中，LLVM安装的目标路径 ${LLVM_INSTALL_PREFIX}
# 确认已安装clang>=15，lld>=15，ccache

LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
```

### 镜像安装

#### 检查镜像版本

**表2** CANN 版本与镜像标签对照表。更多镜像参见 [OVERVIEW.zh.md](../../docker/OVERVIEW.zh.md) 文档。
<table style="table-layout: fixed; width: 100%; border-collapse: collapse;">
  <tr style="height: 50px;">
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">CANN版本</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">芯片类型</th>
    <th style="width: 20%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">Python版本</th>
    <th style="width: 40%; border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f5f5f5;">镜像标签</th>
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

#### 镜像安装

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

# 进入容器
docker exec -u root -it triton-ascend_container /bin/bash
```

## 安装结果验证

安装运行时依赖：

```bash
# 拉取triton-ascend源码仓及用例（可选，非源码编译安装运行示例时需拉源码仓）
git clone https://github.com/triton-lang/triton-ascend.git
cd triton-ascend && pip install -r requirements.txt
```

运行实例：<a href="https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/tutorials/01-vector-add.py" style="text-decoration: none; color: #0066cc;">01-vector-add.py </a>

```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 运行tutorials示例：
python3 ./third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确：

```text
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

## 附录：FAQ<a id = "附录faq" ></a>

### 安装 torch_npu 时出现报错“ERROR: No matching distribution found for torch==2.7.1+cpu”

#### 解决措施

可以尝试手动安装 torch 后再安装 torch_npu：

```bash
pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 编译安装 Triton-Ascend 时，如果GCC < 9.4.0，可能报错 “ld.lld: error: unable to find library -lstdc++fs”

#### 解决措施

一般是链接器无法找到 stdc++fs 库引起的报错。该库用于支持 GCC 9 之前版本的文件系统特性。此时需要手动把 CMake 文件中以下相关代码片段的注释打开。
文件路径：triton-ascend/CMakeLists.txt

```bash
if (NOT WIN32 AND NOT APPLE)
link_libraries(stdc++fs)
endif()
```

### 执行算子时报错 ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'; 'triton._C.libtriton' is not a package

#### 根因分析

 triton-ascend 目录被triton覆盖,导致triton-ascend功能受损。

#### 解决措施

 卸载已损坏的triton-ascend,重新安装即可。以3.2.1 版本为例，可执行如下命令修复：

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

### Triton-Ascend 3.2.1 版本为何新增依赖triton？

答复：Triton-Ascend 是基于Triton进行的二次开发，与Triton安装目录同名。若用户安装Triton-Ascend之后，在此安装triton或依赖triton的三方件，会覆盖triton目录，导致Triton-ascend功能受损。
因此通过增加triton依赖，当triton被覆盖安装时会有如下提醒。

```text
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
triton-ascend 3.2.1 requires triton==3.5.0, but you have triton 3.5.1 which is incompatible.
```

若用户遇到且想恢复triton-ascend功能，可做如下操作：

```bash
pip uninstall triton-ascend triton
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

```

### Triton-Ascend 3.2.1 版本依赖的 Triton 版本为何不一致？

答复：x86 与 arm 使用不同版本的社区 Triton 安装包的原因是社区从 3.5 版本开始才提供 arm 版本安装包：x86 依赖 triton == 3.2.0，arm 依赖 triton == 3.5.0。

### 如何确认芯片类型

您可以使用 npu-smi 命令查看系统上的 NPU 型号。例如，在 npu-smi info 命令的输出中，"910B4" 对应芯片类型 A2 （昇腾 910b 系列）：

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
