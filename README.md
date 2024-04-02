# Using YOLOv7 for crop and weed detection

Used dataset: [The Crop and Weed dataset](https://github.com/cropandweed/cropandweed-dataset)

Used YOLOv7 version: https://github.com/Chris-hughes10/Yolov7-training/issues

--------------

## Installation

--------------

Choose a python version and set a location for the environment

```sh
python3.9 -m venv $TMP/tempenv
source $TMP/tempenv/bin/activate
```

**Optional:** Assure there is a pip in the environment

```sh
python3.9 -m pip install -U pip --no-cache-dir --force-reinstall  
```

[PyTorch-Accelerated:](https://github.com/Chris-hughes10/pytorch-accelerated) is not so well maintained and might downgraded you to a PyTorch version < 2, this installation command prevents the downgrade:

```sh
pip install pytorch-accelerated==0.1.40 --no-dependencies  
```

**Dataset and Model installation:** Installing these can make problems under Windows -> clone them

```sh
# NOTE: Currently my version is not up to date with the original repository, but provided as a (non-proper) submodule here.
# git clone https://github.com/Chris-hughes10/Yolov7-training.git
git clone https://github.com/cropandweed/cropandweed-dataset.git
```

Alternatively my fork:

```sh
https://github.com/Daraan/cropandweed-dataset
```

Note: Note the currently linked yolov7 variant is not compatible with half precision training,
it is possible and you can reach out to me, however, I am not sure if I can currently assist you.

-------------

**Requirements (minimal):**

The code was written with Python 3.9.

The CLI requirements where created by pipreqs and tested the last time this readme was updated (Apr 2024).

```sh
pip install -r requirements_CLI.txt 
```

For the notebook or if you encounter problems you can try with a more restrictive installation, acquired through `pip freeze`:

```sh
pip install -r requirements_complete.txt 
```

**Optional** not needed for this notebook but optionally for supplementary code.
There might be CUDA path problems therefore not putting it into requirements

```sh
pip install deepspeed 
```

### Light version

For the .py files a lighter version

```sh
pip install -r pipreqs_requirements.txt 
```

## Todos

- Create fork for half precision support of Yolov7
- ...
