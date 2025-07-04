#!/bin/bash

# Create a Conda environment named 'gan' with Python 3.10.17
conda create -n gan python=3.10.17 -y

# Initialize Conda (if not already initialized)
# conda init

# # Activate the Conda environment
# conda activate gan

# # Install dependencies from requirements.txt
# if [ -f "requirements.txt" ]; then
#     pip install -r requirements.txt
# else
#     echo "requirements.txt not found. Please ensure it exists in the current directory."
# fi