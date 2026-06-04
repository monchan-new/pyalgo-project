#!/bin/bash
set -e

apt-get update
apt-get upgrade -y
apt-get install -y bzip2 gcc git htop screen vim wget
apt-get clean

# INSTALL MINICONDA
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh
bash Miniconda.sh -b
rm -rf Miniconda.sh

export PATH="/root/miniconda3/bin:$PATH"

# --- conda TOS を自動承認（重要） ---
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
/root/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

# INSTALL PYTHON LIBRARIES
/root/miniconda3/bin/conda install -y pandas
/root/miniconda3/bin/conda install -y ipython
