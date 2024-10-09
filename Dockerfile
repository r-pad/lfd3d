# Use the official Ubuntu 20.04 image as the base
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential libssl-dev zlib1g-dev libbz2-dev \
    git \
    libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
ENV CODING_ROOT="/opt/rpad"

WORKDIR $CODING_ROOT
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="$CODING_ROOT/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install Python 3.10 using pyenv
RUN pyenv install 3.10.0
RUN pyenv global 3.10.0

# Install PyTorch with CUDA support (make sure to adjust this depending on your CUDA version)
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118/

# Make the working directory the home directory
RUN mkdir $CODING_ROOT/code
WORKDIR $CODING_ROOT/code

# Only copy in the source code that is necessary for the dependencies to install
COPY ./src $CODING_ROOT/code/src
COPY ./setup.py $CODING_ROOT/code/setup.py
COPY ./pyproject.toml $CODING_ROOT/code/pyproject.toml
RUN pip install -e .[develop]

# Changes to the configs and scripts will not require a rebuild
COPY ./configs $CODING_ROOT/code/configs
COPY ./scripts $CODING_ROOT/code/scripts

RUN git config --global --add safe.directory /root/code

# Make a data directory.
RUN mkdir $CODING_ROOT/data

# Make a logs directory.
RUN mkdir $CODING_ROOT/logs

# Set up the entry point
CMD ["python", "-c", "import torch; print(torch.cuda.is_available())"]
