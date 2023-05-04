# Using YOLOv7 for crop and weed detection


Used dataset: [The Crop and Weed dataset](https://github.com/cropandweed/cropandweed-dataset)

Used YOLOv7 version: https://github.com/Chris-hughes10/Yolov7-training/issues 


--------------
## Installation
--------------


Choose a python version and set a location for the environment
```
python3.9 -m venv $TMP/tempenv
source $TMP/tempenv/bin/activate
```

Assure there is a pip in the environment
```
python3.9 -m pip install -U pip --no-cache-dir --force-reinstall  
```

This is not so well maintained and might downgraded to a PyTorch version < 2
```
pip install pytorch-accelerated==0.1.40 --no-dependencies  
```

Installing these can make problems under Windows -> clone them
```
git clone https://github.com/Chris-hughes10/Yolov7-training.git
git clone https://github.com/cropandweed/cropandweed-dataset.git
```

Alternatively my fork:
```
https://github.com/Daraan/cropandweed-dataset
```

Note: Note the currently liknked yolov7 variant is not compatible with half precision training.
A fork will be added or contact me.

For the notebook
```
pip install -r requirements.txt 
```

Optional not needed for this notebook but optionally for supplementary code.
There might be CUDA path problems therefore not putting it into requirements
```
pip install deepspeed 
```

### Light version

For the .py files a lighter version
```
pip install -r pipreqs_requirements.txt 
```


## Todos:

- Update readme
- Create fork for half precision support of Yolov7
- ...
