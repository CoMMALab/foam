FROM ubuntu:24.04

LABEL maintainer="Sai Coumar <sai.c.coumar1@gmail.com>"

# Environment variables
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV XDG_RUNTIME_DIR=/tmp/runtime-docker
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV TERM=xterm-256color
ENV PATH="/home/user/bin:${PATH}"

# Set default shell during Docker image build to bash
SHELL ["/bin/bash", "-l", "-c"]

# Update system and install required dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set ARG for GitHub token
ARG GITHUB_TOKEN
ENV GITHUB_TOKEN=${GITHUB_TOKEN}

RUN python3 -m pip config set global.break-system-packages true
RUN pip install \
    contourpy==1.3.1 \
    cycler==0.12.1 \
    fire==0.7.0 \
    fonttools==4.55.3 \
    kiwisolver==1.4.8 \
    matplotlib==3.10.0 \
    numpy==2.2.2 \
    packaging==24.2 \
    pillow==11.1.0 \
    pyglet==1.5.31 \
    pyparsing==3.2.1 \
    python-dateutil==2.9.0.post0 \
    scipy==1.15.1 \
    six==1.17.0 \
    termcolor==2.5.0 \
    trimesh==4.5.3 \
    xmltodict==0.14.2

WORKDIR /foam

# Copy the startup script
COPY startup.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/startup.sh

# Set the startup script as the entrypoint
ENTRYPOINT ["/usr/local/bin/startup.sh"]

CMD ["bash"]