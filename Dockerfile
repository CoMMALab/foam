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
    python3-venv \
    python3-dev \
    libprotobuf-dev \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# uv installs into the system python (no venv), ~10x faster than pip
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv pip install \
    contourpy==1.3.1 \
    cycler==0.12.1 \
    fire==0.7.0 \
    fonttools==4.55.3 \
    kiwisolver==1.4.8 \
    matplotlib==3.10.0 \
    numpy==2.2.2 \
    packaging==24.2 \
    pillow==11.1.0 \
    viser==1.1.1 \
    yourdfpy==0.0.60 \
    pyparsing==3.2.1 \
    python-dateutil==2.9.0.post0 \
    scipy==1.15.1 \
    six==1.17.0 \
    termcolor==2.5.0 \
    trimesh==4.5.3 \
    xmltodict==0.14.2

# Bake the source and the spheretree/manifold binaries into the image, so it runs anywhere without a mount
WORKDIR /foam
COPY . /foam
RUN sed -i 's/\r$//' startup.sh && \
    cmake -Bbuild -GNinja . && cmake --build build/ && uv pip install -e . --no-deps

# Work on the caller's files (mount them at /data); viser tools serve on 8080
WORKDIR /data
ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 8080

# `docker run foam <script> [args]` runs scripts/<script>.py; anything else is executed as-is
ENTRYPOINT ["bash", "/foam/startup.sh"]
CMD ["bash"]
