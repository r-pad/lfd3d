# Use the official Ubuntu 20.04 image as the base
FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CODING_ROOT="/opt/rpad"
ENV PATH="$PATH:/root/.pixi/bin"

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential libssl-dev zlib1g-dev libbz2-dev \
    git \
    libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p $CODING_ROOT/code $CODING_ROOT/data $CODING_ROOT/logs
WORKDIR $CODING_ROOT/code

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

COPY ./pyproject.toml $CODING_ROOT/code/pyproject.toml
COPY ./pixi.lock $CODING_ROOT/code/pixi.lock
COPY ./src $CODING_ROOT/code/src
COPY ./setup.py $CODING_ROOT/code/setup.py
COPY ./configs $CODING_ROOT/code/configs
COPY ./scripts $CODING_ROOT/code/scripts

# Install dependencies using Pixi
RUN pixi install && \
    pixi run install-deps

# Set the default command to test CUDA availability
CMD ["pixi", "run", "test-cuda"]
