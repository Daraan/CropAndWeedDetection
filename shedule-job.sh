#!/bin/bash

cp settings.json ./sbatch-setups/$1

echo "sbatch arguments: ${*:2}"

# shedule batch job and pass copied settings as argument
sbatch ${*:2} --output=./sbatch-setups/$1.out --error=./sbatch-setups/$1.out -J $1 ./test_training.sh ./sbatch-setups/$1