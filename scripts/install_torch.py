import platform
import subprocess
import sys

CUDA_URL = "https://download.pytorch.org/whl/cu130"


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available using nvidia-smi"""
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    # macOS automatically gets MPS-capable build from PyPI
    if platform.system() == "Darwin":
        print("Detected macOS → installing CPU/MPS build from PyPI.")
        return

    # Windows or Linux: check for NVIDIA GPU
    if has_nvidia_gpu():
        print("CUDA GPU detected → installing CUDA wheel.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--index-url", CUDA_URL, "torch", "torchvision", "torchaudio"])
    else:
        print("No CUDA GPU detected → installing CPU build.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio"])


if __name__ == "__main__":
    main()
