#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 10 	
#SBATCH --time=600
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH -J Training

# Copy paste bboxes to ssd

mkdir $TMP/data
# Copy pasint of bboxes is wasted time, many small files
# TODO: try zipping them
#time cp -r $(ws_find industry)/cropandweed-dataset/data/bboxes $TMP/data/bboxes
#time cp -r $HOME/CropAndWeedDetection/cropandweed-dataset/data/bboxes $TMP/data/bboxes

#time cp -r $HOME/CropAndWeedDetection/cropandweed-dataset/data/images $TMP/data/images $TMP/data/images

#
time cp -r $(ws_find industry)/cropandweed-dataset/data/images $TMP/data/images
export CACHE_IMAGES=0
export CACHE_BBOXES=1

# ROOT PATHS
export IMAGES_ROOT=$TMP/data
export BBOX_ROOT=$(ws_find industry)/cropandweed-dataset/data/bboxes

#cd $(ws_find industry)/cropandweed-dataset
source $(ws_find industry)/cropandweed-dataset/workenv/bin/activate
python train.py