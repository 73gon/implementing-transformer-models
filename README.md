## Setup

### 1. Install dependencies
Use Poetry to install all required packages:

```bash
poetry install
```

---

### 2. Configure PyTorch version
Run the helper script to automatically install the correct PyTorch build:

```bash
poetry run python scripts/install_torch.py
```

- On **macOS (M1/M2)** → installs the default CPU/MPS version
- On **Windows (NVIDIA GPU)** → upgrades to the CUDA version

---

That’s it — your environment is ready to use.
