#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 9
#SBATCH --time=780
#SBATCH --mem-per-gpu=64GB
#SBATCH --gres=gpu:1

# help see https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

echo TestTrain
echo "Settings from $1"
#cp settings.json ./sbatch-setups/$1

# Copy paste bboxes to ssd

mkdir $TMP/data
# Copy pasint of bboxes is wasted time, many small files
# TODO: try zipping them

# TEMP
#time cp -r $(ws_find industry)/cropandweed-dataset/data/bboxes $TMP/data/bboxes
#time cp -r $HOME/CropAndWeedDetection/cropandweed-dataset/data/bboxes $TMP/data/bboxes
#export BBOX_ROOT=$TMP/data

# OR WORKSPACE:
export BBOX_ROOT=$(ws_find industry)/cropandweed-dataset/data/

# TEMP uncomment 2
#time cp -r $HOME/CropAndWeedDetection/cropandweed-dataset/data/images $TMP/data/images
#time cp -r $(ws_find industry)/cropandweed-dataset/data/images $TMP/data/images
#export IMAGES_ROOT=$TMP/data

# OR WORKSPACE uncomment 1
export IMAGES_ROOT=$(ws_find industry)/cropandweed-dataset/data/

export CACHE_IMAGES=0
export CACHE_BBOXES=1

# ROOT PATHS
#export IMAGES_ROOT=$(ws_find industry)/cropandweed-dataset/data/

#cd $(ws_find industry)/cropandweed-dataset
source $(ws_find industry)/cropandweed-dataset/workenv/bin/activate
python train.py --settings=$1