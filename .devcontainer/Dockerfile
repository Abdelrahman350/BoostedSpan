# -----------------------------------------------------------------------------
# Stage 1: Builder Stage - Install dependencies and set up the environment
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/cuda-dl-base:25.04-cuda12.9-devel-ubuntu24.04 AS builder

# Build arguments
ARG USERNAME="abdelrahman"
ARG USER_UID=1000
ARG USER_GID=1000

ARG USER_PASSWORD="ubuntu"
ARG ENV_NAME="base-env"
ARG MAINTAINER_EMAIL="eng.abdelrahman.ahmad@gmail.com"

# Set environment variables
ENV TZ="Africa/Cairo" \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Install system dependencies and DVC (from apt repository)
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tree htop sudo wget gdb nano unzip ffmpeg libsm6 libxext6 \
    python3 python3-dev manpages-dev build-essential openssh-client git \
    pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
    cmake curl ca-certificates tar file \
    libosmesa6 libosmesa6-dev freeglut3-dev && \
    # Install DVC via its apt repository
    mkdir -p /etc/apt/keyrings && \
    wget -qO - https://dvc.org/deb/iterative.asc | gpg --dearmor -o /etc/apt/keyrings/packages.iterative.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.iterative.gpg] https://dvc.org/deb/ stable main" \
    | tee /etc/apt/sources.list.d/dvc.list && \
    chmod 644 /etc/apt/keyrings/packages.iterative.gpg /etc/apt/sources.list.d/dvc.list && \
    apt-get update && \
    apt-get install -y dvc && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    -o "/tmp/awscliv2.zip" && \
    unzip /tmp/awscliv2.zip -d /tmp && \
    /tmp/aws/install && \
    rm -rf /tmp/aws /tmp/awscliv2.zip && \
    rm -rf /var/lib/apt/lists/* && \
    apt autoremove -y

# -----------------------------------------------------------------------------
# Create and configure a non-root user
# -----------------------------------------------------------------------------
RUN set -eux; \
    # Remove existing conflicting users/groups
    { [ "$(id -u ubuntu 2>/dev/null)" = "1000" ] && userdel -r ubuntu || true; }; \
    { getent passwd 1000 && userdel -r $(getent passwd 1000 | cut -d: -f1) || true; }; \
    { getent group ${USER_GID} && groupdel $(getent group ${USER_GID} | cut -d: -f1) || true; }; \
    # Create user and group
    groupadd -g ${USER_GID} ${USERNAME}; \
    useradd -u ${USER_UID} -g ${USER_GID} -ms /bin/bash ${USERNAME}; \
    # Configure sudo privileges
    usermod -aG sudo ${USERNAME}; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 0440 /etc/sudoers.d/${USERNAME}; \
    # Set password
    echo "${USERNAME}:${USER_PASSWORD}" | chpasswd; \
    # Create home directory structure
    mkdir -p /home/${USERNAME}/.local /home/${USERNAME}/.ssh; \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}; \
    chmod 700 /home/${USERNAME}/.ssh

# Switch to the non-root user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# -----------------------------------------------------------------------------
# Configure Git for the non-root user
# -----------------------------------------------------------------------------
RUN git config --global user.email "eng.abdelrahman.ahmad@gmail.com" && \
    git config --global user.name "Abdelrahman350" && \
    git config --global init.defaultBranch main

# -----------------------------------------------------------------------------
# uv — PATH must be set before the RUN that installs and calls uv
# -----------------------------------------------------------------------------
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# uv manages its own CPython runtimes — never fall back to the system Python.
# This ensures .venv/bin/python is a proper standalone binary, not a symlink
# to /usr/bin/python3, so venv activation works correctly in every shell.
ENV UV_PYTHON_PREFERENCE="managed"

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv --version && \
    echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc && \
    echo '' >> ~/.bashrc && \
    echo '_uv_auto_venv() {' >> ~/.bashrc && \
    echo '    if [[ -f ".venv/bin/activate" && -z "$VIRTUAL_ENV" ]]; then' >> ~/.bashrc && \
    echo '        source .venv/bin/activate' >> ~/.bashrc && \
    echo '    elif [[ -n "$VIRTUAL_ENV" && ! -f ".venv/bin/activate" ]]; then' >> ~/.bashrc && \
    echo '        deactivate 2>/dev/null || true' >> ~/.bashrc && \
    echo '    elif [[ -z "$VIRTUAL_ENV" && ! -d ".venv" ]] && \' >> ~/.bashrc && \
    echo '         [[ -f "pyproject.toml" || -f "uv.lock" ]]; then' >> ~/.bashrc && \
    echo '        echo "[uv] No .venv found — running uv sync to set up the environment..."' >> ~/.bashrc && \
    echo '        uv sync && source .venv/bin/activate' >> ~/.bashrc && \
    echo '    fi' >> ~/.bashrc && \
    echo '}' >> ~/.bashrc && \
    echo 'export PROMPT_COMMAND="_uv_auto_venv${PROMPT_COMMAND:+;$PROMPT_COMMAND}"' >> ~/.bashrc

WORKDIR /home/mount

# -----------------------------------------------------------------------------
# Stage 2: Final Stage - Attach frequently changing metadata
# -----------------------------------------------------------------------------
FROM builder AS final

LABEL maintainer="${MAINTAINER_EMAIL}" \
    description="Docker image for Generative AI, Computer Vision and Graphics using PyTorch, DVC, UV and Rust (Cargo)" \
    version="1.1"

CMD ["/bin/bash"]