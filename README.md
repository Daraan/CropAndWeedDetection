# Using YOLOv7 for crop and weed detection

Used dataset: [The Crop and Weed dataset](https://github.com/cropandweed/cropandweed-dataset)

Used YOLOv7 version: <https://github.com/Chris-hughes10/Yolov7-training/issues>

## Installation

### Clone the repository

You can download the source code including the models from the latest release.

```sh
# This prevents downloading the large model files; 
# optional, skip this line to include them
export GIT_LFS_SKIP_SMUDGE=1
# Clone the repository and the two submodules
git clone --recurse-submodules https://github.com/Daraan/CropAndWeedDetection.git
```

If you forgot the `--recurse-submodules` you can still download the submodules with:

```sh
# cd CropAndWeedDetection
git submodule init
git submodule update
```

Note: The currently linked Yolov7 variant is not compatible with half precision training.
It is possible, however, I probably cannot assist you in this matter anymore.

--------------

### Create a virtual environment

Choose a python version and set a location for the environment

```sh
python3.9 -m venv env
source env/bin/activate
```

**Optional:** Assure there is a pip in the environment. On my HPC-cluster this was wrong in some cases.

```sh
python3.9 -m pip install -U pip --no-cache-dir --force-reinstall  
```

### Requirements

IMPORANT for WINDOWS users :
Do not install the [cropandweed-dataset](https://github.com/cropandweed/cropandweed-dataset) and [Yolov7-training](https://github.com/Chris-hughes10/Yolov7-training) via `pip`, use the cloned repositories provided through the submodule.

[PyTorch-Accelerated:](https://github.com/Chris-hughes10/pytorch-accelerated) is integrated into the YOLOv7 code but not directly used.
It is not so well maintained and might downgraded you to a PyTorch version < 2, this installation command prevents the downgrade:

```sh
pip install pytorch-accelerated==0.1.40 --no-dependencies  
```

--------------

**Requirements (minimal):**

The code was written with Python 3.9.

The CLI requirements where created by *pipreqs* and tested the last time this in April 2024.

```sh
pip install -r requirements_CLI.txt 
```

For the notebook or if you encounter problems you can try with a more restrictive installation, acquired through `pip freeze`:

```sh
pip install -r requirements_complete.txt 
```

**Optional**: not needed for this notebook, but optionally for supplementary code.
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

- Create fork for half precision support of Yolov7 (#wontfix)
