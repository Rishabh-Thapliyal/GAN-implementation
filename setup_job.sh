#!/bin/bash

# Install Python 3.10.17 with Conda
conda install python=3.10.17 -y

# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install packages from requirements.txt
pip install --no-cache-dir -r requirements.txt

# Verify installations
echo "=== Installed Python Packages ==="
python -m pip list