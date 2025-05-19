FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS builder

# Install the project into `/pyfhd`
WORKDIR /pyfhd

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

# Copy everything over except what's in the .dockerignore
COPY . /pyfhd

# Install the dependencies and project
RUN uv sync --locked --no-dev

# Separate build stage as uv is not needed in the final image
FROM python:3.13-slim-bookworm AS runner

# Install git to get the git commit hash during PyFHD run.
RUN apt-get update && \
    apt-get install --no-install-recommends git -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# This essentially "installs" PyFHD into the final image
COPY --from=builder --chown=app:app /pyfhd /pyfhd

ENV PATH="/pyfhd/.venv/bin:$PATH"

WORKDIR /pyfhd
