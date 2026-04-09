#!/usr/bin/env bash
# exit on error
set -o errexit

# Install uv (modern Python package manager)
# This is needed because the agent uses 'uvx' to run some MCP servers
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install dependencies using pip (from requirements.txt)
pip install -r requirements.txt

# You can also use uv to install if preferred:
# uv pip install -r requirements.txt
