#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies using pip (from requirements.txt)
# This includes 'uv' which we added for 'uvx' support
pip install -r requirements.txt
