# Use Python runtime as the base image
FROM python:3.12.8-slim

# Set the working directory in the container
WORKDIR /app

# Update packages and install libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:0.5.22 /uv /usr/local/bin/uv

# Copy pyproject file
COPY pyproject.toml .

# Install packages
RUN uv sync