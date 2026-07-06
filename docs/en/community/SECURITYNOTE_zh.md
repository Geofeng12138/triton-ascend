# Security Notice

## System Security Hardening

It is recommended to enable ASLR (**Address Space Layout Randomization**) on the system, with level 2 (full randomization mode) preferred. Configuration can be performed as follows:

    echo 2 > /proc/sys/kernel/randomize_va_space

## User Account Recommendations

For security and least privilege considerations, it is not recommended to use Triton-Ascend with administrator accounts such as root.

## File Permission Control

1. Users are advised to implement permission controls and other security measures for sensitive files such as personal private data and commercial assets. It is recommended to set permissions according to the [File Permission Reference](#file-permission-reference).

2. Permission control should be exercised during the installation and usage process. It is recommended to configure permissions based on the [File Permission Reference](#file-permission-reference).

### File Permission Reference

|   Type                             |   Maximum Linux Permission Reference   |
|----------------------------------- |-----------------------|
|  User Home Directory               |   750 (rwxr-x---)     |
|  Program Files (including scripts, libraries, etc.) |   550 (r-xr-x---)     |
|  Program File Directory            |   550 (r-xr-x---)     |
|  Configuration Files               |   640 (rw-r-----)     |
|  Configuration File Directory      |   750 (rwxr-x---)     |
|  Log Files (completed or archived) |   440 (r--r-----)     |
|  Log Files (currently being written)|   640 (rw-r-----)    |
|  Log File Directory                |   750 (rwxr-x---)     |
|  Debug Files                       |   640 (rw-r-----)      |
|  Debug File Directory              |   750 (rwxr-x---)     |
|  Temporary File Directory          |   750 (rwxr-x---)     |
|  Maintenance/Upgrade File Directory|   770 (rwxrwx---)      |
|  Business Data Files               |   640 (rw-r-----)      |
|  Business Data File Directory      |   750 (rwxr-x---)      |
|  Key Components, Private Keys, Certificates, Encrypted File Directory |   700 (rwx------)      |
|  Key Components, Private Keys, Certificates, Encrypted Data |   600 (rw-------)     |
|  Encryption/Decryption Interfaces, Scripts |   500 (r-x------)      |

## Build Security Notice

Triton-Ascend supports source code compilation and installation. During compilation, dependent third-party libraries will be downloaded and build shell scripts executed, generating temporary program files and build directories. Users can manage permissions on files within the source code directory as needed to reduce security risks.

## Public Network Address Notice

The configuration files and scripts of Triton-Ascend contain [public network addresses](#public-network-addresses).

### Public Network Addresses

| Type     | Open Source Code Address                                                                                     | File Name                                      | Public IP Address/Public URL/Domain/Email Address                                                                 | Purpose Description                          |
|----------|------------------------------------------------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------|
| Open Source Import | <https://github.com/triton-lang/triton.git> | .gitmodules | <https://github.com/triton-lang/triton.git> | Triton source code repository address |
| Open Source Import | <https://gitcode.com/Ascend/AscendNPU-IR.git> | .gitmodules | <https://gitcode.com/Ascend/AscendNPU-IR.git> | AscendNPU IR source code repository address |
| Self-developed     | Not applicable                                                                                         | docker/devdocker/setup_triton-ascend_dev.sh | <https://gitcode.com/Ascend/triton-ascend.git>                                                          | Triton-Ascend source code repository address                 |
| Self-developed     | Not applicable                                                                                         | ascend/examples/generalization_cases/run_daily.sh & scripts/prepare_build.sh | <https://gitee.com/shijingchang/triton.git>                                                           | Build dependency code repository                 |
| Self-developed     | Not applicable                                                                                         | setup.py                                   | <https://gitcode.com/Ascend/triton-ascend/>                                                             | Triton-Ascend source code repository address |
| Open Source Import | <https://gitclone.com>                                                            | scripts/prepare_build.sh                   | <https://gitclone.com/github.com/llvm/llvm-project.git>                                               | Dependent LLVM source code repository    |
| Open Source Import | <https://repo.huaweicloud.com>                                            | scripts/prepare_build.sh                           | <https://repo.huaweicloud.com/repository/pypi/simple>                                                | Used to configure pybind11 download link |
| Open Source Import | <https://pypi.tuna.tsinghua.edu.cn>                                                                                         | docker/devdocker/triton-ascend_dev.dockerfile | <https://pypi.tuna.tsinghua.edu.cn/simple>                                                             | Python pip source configuration         |
| Open Source Import | <https://triton-ascend-artifacts.obs.myhuaweicloud.com> | setup.py |`https://triton-ascend-artifacts.obs.myhuaweicloud.com/llvm-builds/{name}.tar.gz` | Used to download precompiled LLVM tools |
| Open Source Import | <https://bootstrap.pypa.io/get-pip.py> | docker/develop_env.dockerfile | <https://bootstrap.pypa.io/get-pip.py> | Used for automated pip installation |
| Open Source Import | <https://llvm.org/LICENSE.txt> | third_party/ascend/include/Dialect/TritonAscend/IR/& third_party/ascend/lib/Dialect/TritonAscend/IR/ | <https://llvm.org/LICENSE.txt> | Apache License link |
| Open Source Import | <https://netlib.org/cephes/> | third_party/ascend/language/cann/libdevice.py | <https://netlib.org/cephes/> | Function source declaration |