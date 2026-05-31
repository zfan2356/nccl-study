# NCCL Study

This repository is a study workspace for NVIDIA NCCL.

## Third-Party Dependencies

- `third-party/nccl`: NVIDIA NCCL, included as a Git submodule from `https://github.com/NVIDIA/nccl.git`.

To clone this repository with submodules:

```bash
git clone --recurse-submodules https://github.com/zfan2356/nccl-study.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```
