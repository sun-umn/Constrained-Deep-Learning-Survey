#!/bin/bash -l        
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=40g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=zhan7867@umn.edu 
#SBATCH -p apollo_agate    
#SBATCH --error=err.out
#SBATCH --output=logs.out
conda activate pygranso003
cd /home/jusun/zhan7867/Deep_Learning_NTR_CST/core
python main.py > pygranso.txt
