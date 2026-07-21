# Release Policy

## Version Numbering

Triton-Ascend follows the [PEP 440](https://peps.python.org/pep-0440/) version specification, with version numbers aligned with the upstream Triton: `vMAJOR.MINOR.PATCH[rcN][.postN]`

- **MAJOR.MINOR**: Corresponds one-to-one with the upstream Triton version. For example, Triton-Ascend `3.2` is based on Triton `3.2`.
- **PATCH**: The `PATCH` version of Triton-Ascend may be higher than the upstream Triton, used for `MAJOR.MINOR` level bug fixes or improvements. For example, both Triton-Ascend `3.2.0` and `3.2.1` are based on Triton `3.2.0`.
- **rcN**: Release candidate versions, released as needed for early community testing and feedback.
- **postN**: Subsequent patches for a released version, released as needed to fix issues in stable versions.

## Branch Strategy

- The `main` branch is the latest development branch, tracking the latest upstream Triton version.
- For each release version, a corresponding release development branch is created (e.g., `release/3.2.x`), sharing the same commit id as the community release.
- Feature development should be carried out in a forked repository and merged into the Triton-Ascend repository via a `PR`.

**`main` Branch Mapping:**

| Triton-Ascend | Triton commit hash                                           | Python    | CANN  | PyTorch | LLVM commit hash                                             | Patch                                                        |
| ------------- | ------------------------------------------------------------ | --------- | ----- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `main`        | [cfc0a9d](https://github.com/triton-lang/triton-ascend/commit/cfc0a9d) | `3.9~3.13` | `9.0.0` | `2.7.1`   | [f6ded0b](https://github.com/llvm/llvm-project/commit/f6ded0b) | [llvm_patch_f6ded0b.patch](https://github.com/triton-lang/triton-ascend/blob/main/third_party/ascend/patch/llvm_patch_f6ded0b.patch) |

## Maintenance Branches and Lifecycle

Maintenance branch statuses include:

- **Active**: Continuously accepts bug fixes, feature improvements, and security patches; will continue to evolve features or release new versions.
- **Maintenance**: Only accepts critical bug fixes and security patches; no further feature improvements will be released.
- **End of Life**: No longer accepts any fixes; branch maintenance has ceased.

| Branch              | Status     | Triton Version | Triton-Ascend Releases              | Maintenance End |
| ----------------- | -------- | ------------ | ----------------------------------- | -------- |
| `main`            | `Active`   | `3.5.0`      | /                                   | /        |
| `release/3.2.1` | `Active`   | `3.2.0`      | `3.2.1`                             | /        |
| `release/3.2.x` | `Maintenance`   | `3.2.0`      | `3.2.0rc2`, `3.2.0rc3`, `3.2.0rc4`, `3.2.0` | /        |

## Release Cadence

- **Stable Versions**: Released according to the project version rhythm; not every upstream Triton version will have a corresponding stable release.
- **rc Versions**: Released in sync with the upstream Triton version rhythm for early user testing.
- **post Versions**: Released as needed to fix issues in existing stable versions.

### Release Timeline

| Date       | Event                     |
| ---------- | ------------------------ |
| 2026-05-06 | Release stable version `3.2.1`     |
| 2026-01-21 | Release stable version `3.2.0`     |
| 2025-11-14 | Release preview version `3.2.0rc4`  |
| 2025-11-12 | Release preview version `3.2.0rc3`  |
| 2025-05-26 | Release preview version `3.2.0rc2`  |

## Version Compatibility Matrix

| Triton-Ascend | Triton | Python              | CANN  | PyTorch | LLVM commit hash | LLVM Patch |
| ------------- | ------ | ------------------- | ----- | ------- | ---------------- | --------- |
| `3.2.1`       | `3.2.0` | `3.9`(x86), `3.10-3.13` | `9.0.0` | `2.7.1`   | `b5cc222`        | -         |
| `3.2.0`       | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc4`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `b5cc222`        | -         |
| `3.2.0rc3`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |
| `3.2.0rc2`    | `3.2.0` | `3.9-3.11`          | `8.5.0` | `2.6.0`   | `86b69c3`        | -         |