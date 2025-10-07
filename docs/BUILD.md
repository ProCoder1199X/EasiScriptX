## EasiScriptX Build Guide

This guide walks you through building EasiScriptX from source on Ubuntu and Windows.

### Prerequisites
- CMake 3.20+
- C++20 compiler
  - Ubuntu: gcc-12+ or clang-15+
  - Windows: MSVC (Visual Studio 2022, Desktop development with C++)
- Git

Optional (feature toggles auto-detected by CMake):
- MPI (OpenMPI or MS-MPI)
- CUDA Toolkit (for GPU)
- ONNX Runtime (C++ API)
- libtorch (PyTorch C++ API)
- Doxygen (docs)

---
### Ubuntu 22.04+

1) Install base tooling
```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config curl doxygen graphviz
```

2) Install dependencies
```bash
# Eigen
sudo apt install -y libeigen3-dev

# Boost
sudo apt install -y libboost-all-dev

# libcurl
sudo apt install -y libcurl4-openssl-dev

# OpenMPI (optional for distributed)
sudo apt install -y libopenmpi-dev openmpi-bin
```

3) Optional accelerators
```bash
# CUDA (optional) - follow NVIDIA docs for your GPU/driver
# ONNX Runtime C++ (optional)
#   See: https://onnxruntime.ai/docs/build/eps.html and download prebuilt C++ package
# PyTorch C++ (libtorch) (optional)
#   See: https://pytorch.org/get-started/locally/ (Select C++/Java, download libtorch)
```

4) Configure and build
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

5) Run tests and examples
```bash
ctest --output-on-failure
./bin/esx ./examples/matmul.esx
```

6) Generate documentation (optional)
```bash
cmake --build . --target docs
xdg-open docs/html/index.html || true
```

---
### Windows 10/11 (MSVC + CMake + VS2022)

1) Install Visual Studio 2022
- Workload: "Desktop development with C++"
- Includes MSVC, CMake, and Ninja

2) Install dependencies
- Eigen: download or use vcpkg: `vcpkg install eigen3`
- Boost: `vcpkg install boost-filesystem boost-system`
- libcurl: `vcpkg install curl`
- Optional: MS-MPI, CUDA, ONNX Runtime, libtorch

Using vcpkg (recommended):
```powershell
git clone https://github.com/microsoft/vcpkg $env:USERPROFILE\vcpkg
$env:VCPKG_ROOT = "$env:USERPROFILE\vcpkg"
$env:PATH += ";$env:VCPKG_ROOT"
& $env:VCPKG_ROOT\bootstrap-vcpkg.bat
vcpkg install eigen3 boost-filesystem boost-system curl --triplet x64-windows
```

3) Configure and build (x64)
```powershell
mkdir build; cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake ..
cmake --build . -j
```

4) Run tests
```powershell
ctest --output-on-failure
```

5) Documentation
```powershell
cmake --build . --target docs
start .\docs\html\index.html
```

---
### Troubleshooting
- If CUDA/ONNX/Torch are not found, the build will continue with those features disabled.
- Ensure that environment variables for optional SDKs are set (e.g., `Torch_DIR`, `onnxruntime_DIR`).
- On Windows with vcpkg, keep the same `-DCMAKE_TOOLCHAIN_FILE` for all configure/build invocations.


