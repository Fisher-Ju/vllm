# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Architecture Overview

vLLM is a high-throughput LLM inference and serving engine. The V1 architecture uses a multi-process design:

| Process Type | Count | Purpose |
| - | - | - |
| API Server | `A` (default `DP`) | HTTP requests, input processing |
| Engine Core | `DP` | Scheduler, KV cache management |
| GPU Worker | `N` (= `DP x PP x TP`) | One per GPU, executes forward passes |
| DP Coordinator | 1 if `DP > 1` | Load balancing across DP ranks |

### Key Directories

- `vllm/v1/engine/` - Engine core (core.py, async_llm.py), scheduler, KV cache
- `vllm/v1/executor/` - Process orchestration (multiproc_executor.py, ray_executor.py)
- `vllm/v1/worker/` - GPU workers (gpu_worker.py), model runners (gpu_model_runner.py)
- `vllm/model_executor/models/` - Model implementations (200+ architectures)
- `vllm/model_executor/layers/` - Common layers (attention, linear, quantization)
- `vllm/entrypoints/` - API servers (openai/), CLI (cli/), offline inference (llm.py)
- `csrc/` - CUDA kernels (attention, moe, quantization)
- `tests/` - Organized by component (v1/, kernels/, models/, distributed/)

### Class Hierarchy

```
Engine → Worker → ModelRunner → Model (torch.nn.Module)
```

Each level receives a `VllmConfig` object containing all engine-level configuration.

## Key Implementation Patterns

### VllmConfig Pattern

All classes accept `VllmConfig` - the engine-level global state. Model constructors use:
```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```
This enables extensibility, uniformity, and efficient sharding/quantization at initialization.

### Forward Pass Flow (V1)

```
Engine Core (scheduler) → Worker → ModelRunner → Model
```
The scheduler picks requests, workers execute forward passes, model runners prepare tensors.

### Model Registration

Models are registered in `vllm/model_executor/models/__init__.py` mapping architecture names to classes. The registry is queried via the `architectures` field from HuggingFace config.json.

### Custom Operations (CUDA Kernels)

Custom ops require:
1. Schema registration following PyTorch guidelines
2. Meta-functions for dynamic dims (Python)
3. `torch.library.opcheck()` testing - see `tests/kernels/utils.py` for the `opcheck` helper

## Domain-Specific Guides

Read these before modifying related code:

- **Models**: `docs/contributing/model/` - model registration, multimodal support, testing
- **Kernels**: `docs/contributing/README.md` (section "Adding or Changing Kernels") - custom ops registration, meta-functions, opcheck
- **Architecture**: `docs/design/arch_overview.md` - class hierarchy, process architecture
- **HuggingFace Integration**: `docs/design/huggingface_integration.md` - model loading flow

## Development Commands

### Environment Setup

```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate

# Install lint tools
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing vLLM

```bash
# Python-only changes (uses precompiled kernels)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# C++/CUDA changes (requires full build)
uv pip install -e . --torch-backend=auto --no-build-isolation
```

### Running Tests

```bash
# Install test dependencies
uv pip install -r requirements/test/cuda.in  # resolves for current platform
# or on x86_64: uv pip install -r requirements/test/cuda.txt

# Run specific test file (Linux/Mac)
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Run specific test file (Windows)
.venv\Scripts\python.exe -m pytest tests\path\to\test_file.py -v

# Run all tests in a directory
.venv/bin/python -m pytest tests/v1/engine/ -v
```

### Running Linters

```bash
# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff-check --all-files

# Run mypy as in CI
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

### Documentation

```bash
uv pip install -r requirements/docs.txt
mkdocs serve                           # with API ref (~10 min)
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # API ref off (~15 sec)
```

## PR Conventions

Prefix titles with: `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[Hardware][Vendor]`, `[Misc]`

For model changes: include model name in title (e.g., `[Model] Llama`).