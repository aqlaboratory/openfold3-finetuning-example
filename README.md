# Finetuning Dataset Generation

## Setup

Install this repo and OpenFold3 from source. This script relies on imports from the OpenFold3 repo. 
```
git clone git@github.com:aqlaboratory/openfold3-finetuning-example.git
# clone OF3
git clone git@github.com:aqlaboratory/openfold-3.git
cd openfold-3

conda create -n of3-internal python=3.13 -y
conda activate of3-internal

pip install -e .

setup_openfold
```
