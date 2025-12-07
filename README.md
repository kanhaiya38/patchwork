# Patchwork-L: LoRA-based Delta Compression for Continual LLMs on Edge Devices

## Overview

This repository provides the codebase and evaluation pipeline for the associated research project. To ensure reproducibility, follow the setup instructions below in the specified order.

## Dataset

The project uses the **TRACE Benchmark** dataset. The dataset is publicly available through the following sources:

* GitHub repository: [https://github.com/BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE)
* Direct download link: [https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view](https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view)

If downloading via command line, `gdown` can be used.

### Downloading with `gdown`

Install `gdown`:

```bash
pip install gdown
```

Download the dataset:

```bash
gdown "https://drive.google.com/uc?id=1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
```

Unzip the benchmark archive:

```bash
unzip ./TRACE-Benchmark.zip
```

Ensure the dataset directory remains in the expected location referenced by the evaluation scripts.

## Environment Setup

A Conda environment specification is provided to ensure consistent dependencies across systems. Create and activate the environment as follows:

```bash
conda env create -f environment.yml
conda activate smr
```

It is recommended to use the provided `environment.yml` without modification to avoid dependency conflicts.

## Running the Evaluation

After downloading the dataset and configuring the environment, execute the evaluation script:
Read the `scripts/test.sh` for more information.

```bash
bash scripts/test.sh
```

This script assumes default dataset paths and environment configuration. If you alter the directory structure, update the script accordingly.

## Additional Notes

* A GPU-enabled system is recommended for efficient execution.
* Tested on CUDA Version 12.9
