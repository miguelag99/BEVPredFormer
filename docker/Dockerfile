# Stage 1: Base image with system dependencies
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base
# FROM nvcr.io/nvidia/pytorch:24.10-py3 AS base

# Set environment variables
ENV FORCE_CUDA="1" 
ENV TZ=Europe/Madrid

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Stage 2: System dependencies and user setup
FROM base AS system-deps

# Obtain the UID and GID of the current user
ARG USER
ARG USER_ID
ARG USER_GID

RUN groupadd -g $USER_GID $USER \
    && useradd --uid $USER_ID --gid $USER_GID -m $USER \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Install system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    mesa-utils \
    libglapi-mesa \
    libqt5gui5 \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    wget \
    figlet \
    jupyter \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Stage 3: Python dependencies
FROM system-deps AS python-builds

# Set the user
ARG USER
USER $USER

# Upgrade pip
RUN python -m pip install --upgrade pip
ENV PATH="${PATH}:/home/$USER/.local/bin"

# Install MMCV from source
WORKDIR /home/$USER
RUN git clone -b v2.2.0 https://github.com/open-mmlab/mmcv.git
WORKDIR /home/$USER/mmcv
RUN pip install -e . -v

# Stage 4: Project-specific dependencies
FROM python-builds AS project-build
ARG USER

# Install requirements
RUN mkdir -p /home/$USER/.cache
COPY requirements.txt /home/$USER/.cache
RUN pip install -r /home/$USER/.cache/requirements.txt

# Disable Jupyter authentication
# RUN jupyter notebook --generate-config
# RUN echo "c.NotebookApp.token = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py
# RUN echo "c.NotebookApp.password = ''" >> /home/$USER/.jupyter/jupyter_notebook_config.py

# Set working directories
WORKDIR /home/$USER/Datasets
WORKDIR /home/$USER/workspace

# Set custom commands, aliases, and environment variables
ENV USER=$USER
RUN echo "alias PRUEBAS=\"cd /home/$USER/workspace && python bevpredformer/pruebas.py\"" >> /home/$USER/.bashrc

# Change terminal color
ENV TERM=xterm-256color
RUN echo "PS1='\[\e[91m\]\u\[\e[0m\]@\[\e[93m\]\h\[\e[0m\]:\[\e[35m\]\w\[\e[0m\] > '" >> ~/.bashrc
RUN echo "source ~/.bashrc" >> ~/.bash_profile

# Entrypoint
ENTRYPOINT [ "docker/entrypoint.sh" ]
# ENTRYPOINT [ "/bin/bash", "-l", "-c"  ]