FROM rust:1-bookworm

# Install build and OpenCL headers (compile-time), BLAS/LAPACK, git, curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        git \
        curl \
        ocl-icd-opencl-dev \
        clinfo \
        libblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Set a default workdir used by Dev Containers / Codespaces
WORKDIR /workspaces/ha-ndarray

