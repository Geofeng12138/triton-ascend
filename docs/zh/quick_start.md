# 快速入门

## 项目介绍

Triton-Ascend 是适配华为 Ascend 昇腾芯片的 Triton 优化版本，用于高效进行核函数自动调优、算子编译及部署，通过兼容 Triton 核心语法并针对昇腾 NPU 特性进行深度优化，能够帮助用户在昇腾平台上快速开发和部署高性能计算任务。

## 软件包安装
本文以 Ubuntu 22.04 环境下通过软件包部署方式运行向量加法示例为例，指导用户快速上手使用 Triton-Ascend。如需体验更多安装方式请阅读[安装指南](./installation_guide.md)文档。

### 环境准备

#### 硬件要求

支持的操作系统: linux（aarch64/x86_64）

支持的 Ascend 产品: Atlas A2/A3/A5 系列

最小硬件配置: 单卡 32GB 内存（推荐）

#### 软件依赖

确定 Python 和 CANN 软件版本并安装。更多配套关系请参考安装指南的[产品版本配套说明](./installation_guide.md#安装依赖)。

- Python 版本选择：py3.9-py3.11 均可。

- CANN 版本选择：可以访问昇腾社区官网，根据其提供的<a href="https://www.hiascend.com/cann/download" style="text-decoration: none; color: #0066cc;">社区软件安装指引</a>完成 CANN 的安装与配置。建议下载安装 9.0.0 版本。

### 具体实施（以whl包安装为例）

```bash
# 以安装 triton-ascend 3.2.1 为例
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

## 快速开始

**运行 tutorials 中向量加法示例验证结果**

向量加法实例：[01-vector-add.py](../../third_party/ascend/tutorials/01-vector-add.py)
通过对比 Triton 核函数与 PyTorch 原生计算的输出结果进行对比，证明昇腾 NPU 设备可正确调用 Triton 核函数并保证计算精度。

```bash
# 设置CANN环境变量（以root用户默认安装路径`/usr/local/Ascend`为例）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 拉取triton-ascend源码仓及用例（可选，非源码编译安装运行示例时需拉源码仓）
git clone https://github.com/triton-lang/triton-ascend.git
# 运行tutorials示例：
python3 ./triton-ascend/third_party/ascend/tutorials/01-vector-add.py
```

观察到类似的输出即说明环境配置正确。

```shell
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```
