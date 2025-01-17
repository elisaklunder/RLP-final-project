#!/bin/bash
#SBATCH --time=01:40:00
#SBATCH --job-name=hyperparam
#SBATCH --mem=8000

module purge

module load Python/3.9.6-GCCcore-11.2.0
pip install -U pip
pip install -q swig
pip install -r requirements.txt

python agents/hyperparams.py
