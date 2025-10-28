import os, platform, subprocess, sys

CUDA_URL = "https://download.pytorch.org/whl/cu124"

def main():
    # macOS automatically gets MPS-capable build from PyPI
    if platform.system() == "Darwin":
        print("Detected macOS → installing CPU/MPS build from PyPI.")
        return

    # Windows or Linux: check for NVIDIA GPU
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA GPU detected → upgrading to CUDA wheel.")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade",
                "--index-url", CUDA_URL,
                "torch", "torchvision", "torchaudio"
            ])
        else:
            print("No CUDA GPU detected → keeping CPU build.")
    except Exception as e:
        print("Torch not installed yet:", e)

if __name__ == "__main__":
    main()
