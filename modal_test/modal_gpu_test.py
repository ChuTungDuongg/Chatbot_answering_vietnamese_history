import modal

app = modal.App("vn-history-gpu-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch")
)


@app.function(image=image, gpu="L4")
def gpu_test():
    import subprocess
    import torch

    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", round(props.total_memory / 1024**3, 2), "GB")
        print("PyTorch:", torch.__version__)
        print("CUDA runtime:", torch.version.cuda)

    subprocess.run(["nvidia-smi"], check=True)


@app.local_entrypoint()
def main():
    gpu_test.remote()