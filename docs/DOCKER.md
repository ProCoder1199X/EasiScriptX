## EasiScriptX Docker Usage

We provide two images: CPU-only and CUDA-enabled.

### Build locally
```bash
# CPU
docker build -t easiscriptx:cpu -f docker/Dockerfile.cpu .

# CUDA (requires NVIDIA Docker runtime)
docker build -t easiscriptx:cuda -f docker/Dockerfile.cuda .
```

### Run
```bash
# CPU
docker run --rm -it -v $PWD:/workspace easiscriptx:cpu esx /workspace/examples/matmul.esx

# CUDA
docker run --rm -it --gpus all -v $PWD:/workspace easiscriptx:cuda esx /workspace/examples/matmul.esx
```

Prebuilt images (optional) can be published to Docker Hub/GHCR. Update tags and CI accordingly.


