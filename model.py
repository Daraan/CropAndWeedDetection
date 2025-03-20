import sys
import os
import glob
import time
from functools import wraps
from tqdm import tqdm
import numpy as np

from typing import Union, List, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchmetrics

import pytorch_lightning as pl

try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
except ModuleNotFoundError as e:
    print(e, "\n")


from PIL import Image
import cv2
import matplotlib.pyplot as plt

import albumentations as A

BACKUP_SEED = 4222  # In case non is set

# Choose path where the github repository will be cloned to an the cnw folder will be
CNW_PATH = "./cropandweed-dataset/"
OPTUNA_DB_PATH = "./OptunaTrials/trials.db"
MODEL_PATH = os.environ.get("MODEL_PATH", "./models")


# Path were the downloaded files images, segmentation masks, ... will be
if os.getcwd().startswith("/ceph") or os.getcwd().startswith("/pfs"):  # dws or bw server
    DATA_PATH = os.path.join(CNW_PATH, "data")
else:
    DATA_PATH = r"./data"

IMAGES_PATH = os.path.join(DATA_PATH, "images")

if CNW_PATH not in sys.path:
    sys.path.append(CNW_PATH)  # folder contains hypen so using this workaround
    sys.path.append(CNW_PATH + "/cnw")

import cnw
from cnw.utilities.datasets import DATASETS

import utils

# fore developement
from utils import visualize_bbox

sys.path.append("./Yolov7-training")
# from ultralytics.yolo.utils.loss import BboxLoss
from yolov7 import create_yolov7_model, create_yolov7_loss
from yolov7.trainer import filter_eval_predictions
from yolov7.models.yolo import Yolov7Model

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ChainedScheduler,
    ReduceLROnPlateau,
    SequentialLR,
    OneCycleLR,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision

BBOX_PARAMS = A.BboxParams(
    format="pascal_voc",  # Left, Top, Right, Bottom
    # format="yolo",
    label_fields=["class_labels"],
    min_area=30,  # Min amount of pixels
    min_visibility=0.2,
)  # area ratio from after to the before

##################


class CropDataset(torchvision.datasets.VisionDataset):
    """
    Dataset Interface to load images and apply augmentations.
    Allows to use albumination transformations in parallel for mask and images
    """

    # target_transform : A.Compose
    # transforms : torchvision.transforms.Compose

    # TODO if stacking, downscale before stacking.

    def __init__(
        self,
        root=DATA_PATH,
        *,
        dataset,
        stage: Union["test", "valiation", "train"],
        image_size,
        stack2_images,
        train_val_test_indices: Dict[str, List[int]] = None,  # manually choose which indices to use;
        train_val_test_ratio=(0.7, 0.15, 0.15),
        transforms=None,
        transform=None,
        target_transform=None,
        normalize_images=False,
        half_precision=False,
        cache_bboxes=False,
        cache_images=False,
        _init_files=True,  # Set files by dataloader globally, for less overhead.
        seed=None,
    ):  # should be set in the transformations!):
        """
        From documentation:
        transforms (callable, optional) – A function/transforms that takes in an image
            and a label and returns the transformed versions of both.

        transform (callable, optional) – A function/transform that takes in an PIL image
            and returns a transformed version. E.g, transforms.RandomCrop

        target_transform (callable, optional) – A function/transform that takes in
            the target and transforms it.
        """
        super().__init__(root, transforms, transform, target_transform)
        self.dataset = dataset
        self.stage = stage
        self.normalize_images = normalize_images
        self.data_dir = root
        self.try_bbox_caching = cache_bboxes
        self.try_image_caching = cache_images

        if isinstance(root, dict):
            self.image_root = root["images"]
            self.bbox_root = root["bboxes"]
        else:
            self.image_root = root
            self.bbox_root = root

        self.stack2_images = stack2_images
        self.image_size = image_size
        self.half_precision = half_precision
        self.downscale_transform = A.Compose(
            [
                A.SmallestMaxSize(max(image_size)),
                A.RandomSizedCrop(
                    min_max_height=(max(image_size), max(image_size)),
                    w2h_ratio=1.0,
                    height=max(image_size),
                    width=max(image_size),
                ),
            ],
            bbox_params=BBOX_PARAMS,
        )

        # For perfomance and integrity this allows to set the specific values
        # outside of the initialization.
        # For example the indices can be assigned manually to be 100% sure
        # that train, val, test data do not overlapp.
        if _init_files:
            image_dir = os.path.join(self.image_root, "images")
            # If stage val, train load eval boxes
            if stage == "train":
                bbox_dir = os.path.join(self.bbox_root, "bboxes", dataset)
            else:
                bbox_dir = os.path.join(
                    self.bbox_root, "bboxes", dataset + "Eval"
                )  # allows more lenient preidctions for validation

            # TODO: need to change validation vegetation label id?
            bbox_dir = os.path.join(self.bbox_root, "bboxes", dataset)
            # Extra data sadly unused
            # self.annotation_dir = os.path.join(root, 'Annotations')

            # Some overhead storage. Can be removed later
            # This could be a view!
            # self.all_masks = self.get_files() # masks are filtered for the subsets
            self.all_bboxes = self.get_files()  # masks are filtered for the subsets
            # Filter images based on stage TODO ??
            self.all_images = np.asarray(
                [os.path.join(image_dir, os.path.basename(file)[:-4] + ".jpg") for file in self.all_bboxes], dtype=str
            )
            # verify that to each mask an image exist
            # for i in self.all_images:
            #    if not os.path.exists(i):
            #        raise FileNotFoundError(i)

            # Perform train_val_test split and get indices
            if train_val_test_indices is None:
                self.indices = utils.train_val_test_split(self.all_images, seed or BACKUP_SEED, train_val_test_ratio)
                self.images = self.all_images[self.indices[self.stage]]
                # self.masks = self.all_masks[self.indices[self.stage]]
            else:
                self.indices = train_val_test_indices
                self.images = self.all_images[self.indices[self.stage]]  # This should/could be a view
                # self.masks = self.all_masks[self.indices[self.stage]]

            # self.bboxes =  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.masks ]
            self.bboxes = self.all_bboxes[self.indices[self.stage]]
            _init_caches(self)

    def _init_caches(self):
        try:
            if self.try_bbox_caching:
                self.cache_bboxes()
            self.bboxes_cached = self.try_bbox_caching  # True or False
        except Exception as e:
            self.bboxes_cached = False
            raise

        try:
            if self.try_image_caching:
                self.cache_images()
            self.images_cached = self.try_image_caching  # True or False
        except Exception as e:
            self.images_cached = False

    def get_all_bboxes(self):
        """Loads all bboxes into an array. NOTE: This function might not be up do date"""
        print("Loading bboxes")
        bbox_dir = os.path.join(self.bbox_root, "bboxes", self.dataset)
        # This should be quite fast but sometimes it seams to stuck at certain files.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # supresses warnings of empty files
            # .csv is pascal voc format# NOTE: HARDCODED
            files = (os.path.join(bbox_dir, os.path.basename(file)[:-4] + ".csv") for file in self.bboxes)
            bboxes = []
            for file in tqdm(files, total=len(self.bboxes)):
                bboxes.append(np.loadtxt(file, delimiter=",", dtype=int))
        return np.array(bboxes, dtype=object)

    def get_files(self, path=None):
        """Find images belonging to the chosen dataset via the bbox.csv files."""
        path = path or self.data_dir
        if isinstance(path, dict):
            path = path["bboxes"]
        files = np.array(glob.glob(os.path.join(path, "bboxes", self.dataset, "*.csv")))
        if len(files) == 0:
            print("path:", path)
            raise ValueError("No files found in", os.path.join(os.path.join(path, "bboxes", self.dataset)))
        return files

    @staticmethod
    def get_class_label(label_id):
        """Get the class name from a predicted label id"""
        return DATASETS["CropAndWeed"].get_label_name(label_id)

    def get_bbox_color(self, label_id):
        return DATASETS[self.settings["dataset"]["name"]].get_label_color(label_id)
        mapped_id = DATASETS[DATASET].get_mapped_id(label_id)
        return DATASETS[DATASET].get_label_color(mapped_id)

    def __len__(self):
        return len(self.indices[self.stage])

    # Evaluate during compilation
    if BBOX_PARAMS.format == "pascal_voc":

        @staticmethod
        def load_bbox(path):
            bbox = np.loadtxt(path, delimiter=",", dtype=np.int32).reshape(-1, 7)
            return bbox[:, 4], bbox[:, :4]  # label, bbox
    else:

        @staticmethod
        def load_bbox(path):
            bbox = np.loadtxt(path, delimiter=" ", dtype=np.float32).reshape(
                -1, 5
            )  # yolo format, assure not (5,) shaped
            return bbox[:, 0], bbox[:, 1:]

    def cache_bboxes(self):
        """caches all images and labels"""
        # This might take some time on servers because files need to be transfered
        # TODO could be parallies and then joined.
        all_bboxes = []
        all_labels = []
        start = time.perf_counter()
        for bbox_file in self.bboxes:
            labels, bbox = self.load_bbox(bbox_file)
            all_bboxes.append(bbox)
            all_labels.append(labels)
        self.cached_bboxes = np.array(all_bboxes, dtype=object)
        self.cached_labels = np.array(all_labels, dtype=object)
        print("BBoxes cached in", format(time.perf_counter() - start, ".1"), "s")

    def cache_images(self):
        """Loads all images into RAM"""
        # TODO: Should be parallized
        print("Loading and caching all images - this will need a lot of RAM.")
        print("Trying to allocate space...")
        image = cv2.imread(self.images[0])
        image_stack = np.empty((len(self.images), *image.shape), dtype=float)
        print(
            "Allocated array of size", image_stack.shape, format(image_stack.nbytes / 1024 / 1024 / 1024, ".4f"), "GB"
        )
        image_stack[0] = image
        try:
            for idx, file in enumerate(tqdm(self.images[1:]), 1):
                image_stack[idx] = cv2.imread(self.images[idx])
        except BaseException as e:
            print("WARNING: Images not cached")
            raise
        self.cached_images = image_stack

    # Would be great to differentiate between at compile time but in the current setup not possible
    # TODO: make aa new class or overwrite __getitem__ on the class (does not work on instances)
    def __getitem__(self, index):
        return self._getitem_stacked(index) if self.stack2_images else self._getitem_normal(index)

    def _getitem_stacked(self, index):
        """
        Stack two images vertically together by chosing a second one randomly.
        Advantage no blank color padding needed, more variation and more bounding boxes per image.
        """
        if self.images_cached:
            image = self.cached_images[index].copy()
        else:
            image = cv2.imread(self.images[index])
        if self.bboxes_cached:
            labels = self.cached_labels[index].copy()
            bbox = self.cached_bboxes[index].copy()
        else:
            bboxf = self.bboxes[index]  # No disc read
            labels, bbox = self.load_bbox(bboxf)

        if self.stage != "train":
            # Return without stacking two images, transform should be resizing&ToTensor
            transformed = self.target_transform(
                image=np.asarray(image, np.float32) / 255,  # bring into 0...1 range
                # mask=mask,
                bboxes=bbox,
                class_labels=labels,
            )
            return self._format_transformed(transformed, index)

        # Load 2nd image to patch together
        index2 = np.random.choice(len(self.images))
        if self.images_cached:
            image2 = self.cached_images[index2].copy()
        else:
            image2 = cv2.imread(self.images[index2])
        if self.bboxes_cached:
            bbox2 = self.cached_bboxes[index2].copy()
            labels2 = self.cached_labels[index2].copy()
        else:
            bbox2f = self.bboxes[index2]
            labels2, bbox2 = self.load_bbox(bbox2f)

        if BBOX_PARAMS.format == "pascal_voc":
            bbox2[:, [1, 3]] += image.shape[0]  # Adjust Y values of 1st image
        else:
            raise NotImplementedError("Non pascal_voc format")

        full_img = np.vstack((image, image2)).astype(np.float32) / 255  # NOTE: numpy 1.24 allows dtype for vstack
        del image, image2
        bboxes = np.vstack([bbox, bbox2])
        # Downscale
        down_transforms = self.downscale_transform(
            image=full_img, bboxes=np.vstack([bbox, bbox2]), class_labels=np.hstack([labels, labels2])
        )
        del full_img
        # TODO join transforms else overhead in bbox format conversions
        transformed = self.target_transform(**down_transforms)
        return self._format_transformed(transformed, index)

    def _getitem_normal(self, index):
        """
        Bounding box logic: Left, Top, Right, Bottom, Label ID, Stem X, Stem Y
        """
        image = cv2.imread(self.images[index])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Theoretically RGB not necessary, some overhead
        # SEGMENTATION:
        # mask  = cv2.imread(self.masks[index], cv2.IMREAD_UNCHANGED)
        bboxf = self.bboxes[index]  # No disc read
        labels, bbox = self.load_bbox(bboxf)

        if self.target_transform is not None:
            try:
                transformed = self.target_transform(
                    image=np.asarray(image, np.float32) / 255,  # bring into 0...1 range
                    # mask=mask,
                    bboxes=bbox,
                    class_labels=labels,
                )
            except:
                print(bboxf, bbox)
                raise
            image = transformed["image"][2, 1, 0]  # now in tensor format CxHxW, swap to RGB
            # mask  = transformed["mask"]
            bbox = transformed["bboxes"]
            labels = transformed["class_labels"]
            # if mask.dtype != torch.int64:
            #    mask = mask.to(torch.int64)
            return self._format_transformed(transformed, index)
        return self._format_transformed({"image": image, "bboxes": bboxes, "class_labels": labels}, index)

    def _format_transformed(self, transformed, index):
        image = transformed["image"][[2, 1, 0]]  # Swap channel order cv2.COLOR_BGR2RGB
        # mask  = transformed["mask"]
        bbox = transformed["bboxes"]
        labels = transformed["class_labels"]

        if len(bbox) > 0:
            # bboxes to yolo format
            if BBOX_PARAMS.format == "pascal_voc":
                boxes = torchvision.ops.box_convert(
                    torch.as_tensor(bbox, dtype=torch.float32),
                    "xyxy",
                    "cxcywh",  #  if not self.half_precision else torch.float16
                )
                boxes[:, [1, 3]] /= self.image_size[0]  # normalized height 0-1
                boxes[:, [0, 2]] /= self.image_size[1]  # normalized width 0-1
            else:
                boxes = bbox
            classes = np.expand_dims(labels, 1)
            labels_out = torch.hstack(
                (
                    torch.zeros((len(boxes), 1)),
                    torch.as_tensor(classes, dtype=torch.float32),  #  if not self.half_precision else torch.float16
                    boxes,
                )
            )
        else:
            labels_out = torch.zeros((0, 6))

        # if self.half_precision:
        #    image = image.to(torch.float16)
        #    labels_out = labels_out.to(torch.float16)
        return (
            image,
            labels_out,
            torch.as_tensor(index),
            torch.as_tensor(self.image_size),
        )  # height x width (array format)

    @staticmethod
    def bbox_collate(batch):
        # torch expects that the contents of the batch have the same length
        # this behaviour is overwritten here.
        # As there can be multiple bounding boxes, this has to be handled here
        # additionally each bounding box (in labels) will get an image_id assigned
        # todo: add (or reason why not to add them, parallelization issues?) indices instead
        images, labels, indices, image_sizes = zip(*batch)

        for i, l in enumerate(labels):
            l[:, 0] = i  # add target image index for build_targets() in loss fn
        return (
            torch.stack(images, 0),
            torch.cat(labels, 0),
            torch.stack(indices, 0),
            torch.stack(image_sizes, 0),
        )


class CropAndWeedDataModule(pl.LightningDataModule):
    """
    From original documentation:
    "A DataModule standardizes the training, val, test splits, data preparation and transforms. The main
    advantage is consistent data splits, data preparation and transforms across models."
    """

    def __init__(
        self,
        dataset,
        data_dir,
        *,
        seed=BACKUP_SEED,
        eval_batch_scale=2,
        batch_size,
        num_workers,
        image_size,
        train_transform,
        test_transform,
        stack2_images,
        train_val_test_ratio=[0.7, 0.15, 0.15],
        half_precision=False,
        cache_images=False,
        cache_bboxes=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.indices = None
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.stack2_images = stack2_images
        self.image_size = image_size
        self.half_precision = half_precision

        self.cache_images = cache_images
        self.cache_bboxes = cache_bboxes

        self.batch_size = batch_size
        self.eval_batch_scale = eval_batch_scale
        self.num_workers = num_workers
        # In case bbox and images are located at different locations
        if isinstance(data_dir, dict):
            self.image_root = data_dir["images"]
            self.bbox_root = data_dir["bboxes"]
        else:
            self.image_root = data_dir
            self.bbox_root = data_dir
        self.data_dir = data_dir
        self.train_val_test_ratio = train_val_test_ratio
        self.seed = seed or BACKUP_SEED

    def prepare_data(self, repository_path=CNW_PATH, *, with_mapping=True, check_exists=True):
        """NOTE: Dowloads ~10GB and mapping annotations will also take a while."""
        # Todo currently no customization.
        if check_exists:
            # Custom paths do exist (e.g a SSD and HDD) check them
            if isinstance(self.data_dir, dict):
                # Same path we can check:
                if os.path.exists(self.data_dir["bboxes"]) and os.path.exists(self.data_dir["images"]):
                    print(
                        "Data paths already exists. Skipping download and mask generation. No integrity check was performed."
                    )
                    return
                # One path does not exist
                # Check if it the same path (we can handle) else raise error
                if self.data_dir["bboxes"] != self.data_dir["images"]:
                    raise FileNotFoundError(
                        "Image and BBox path differ but one does not exist. Please specify one path only to download files or check the passed paths",
                        self.data_dir["bboxes"],
                        self.data_dir["images"],
                    )
                # Both paths are the same but does not exist -> download files
            # Only one path passed as path like object
            elif os.path.exists(self.data_dir):
                print(
                    "Data path already exists. Skipping download and mask generation. No integrity check was performed."
                )
                return
        print("Cloning github repository for data acquisition")
        os.system(" ".join(["git clone", "https://github.com/cropandweed/cropandweed-dataset.git", repository_path]))
        print("Downloading data. This can take a while.")
        from cnw import setup

        if isinstance(self.data_dir, str):
            setup.setup(self.data_dir, with_mapping)
        else:  # dict of paths
            print("Print ignoring data setup for now")

    def setup(self, stage: str = "fit"):
        # Gather files and do train test val split
        if self.indices is None:
            self.all_bboxes = CropDataset.get_files(self, path=self.bbox_root)  # bbox/dataset will be appended
            image_dir = os.path.join(self.image_root, "images")
            self.all_images = np.asarray(
                [os.path.join(image_dir, os.path.basename(file)[:-4] + ".jpg") for file in self.all_bboxes], dtype=str
            )

            # uncommented for performance
            # verify that to each mask an image exist
            # for i in self.all_images:
            #    if not os.path.exists(i):
            #        raise FileNotFoundError(i)

            # Perform train_val_test split and get indices
            self.indices = utils.train_val_test_split(
                self.all_images, self.seed or BACKUP_SEED, self.train_val_test_ratio
            )

        # bbox_dir = os.path.join(self.bbox_root, 'bboxes', self.dataset)
        # bbox_dir_eval = os.path.join(self.bbox_root, 'bboxes', self.dataset+"Eval")

        if stage == "fit" or stage is None:
            if not hasattr(self, "data_train"):
                self.data_train = CropDataset(
                    self.data_dir,
                    dataset=self.dataset,
                    stage="train",
                    stack2_images=self.stack2_images,
                    image_size=self.image_size,
                    train_val_test_ratio=self.train_val_test_ratio,
                    seed=self.seed,
                    target_transform=self.train_transform,
                    half_precision=self.half_precision,
                    cache_bboxes=self.cache_bboxes,
                    cache_images=self.cache_images,
                    _init_files=False,
                )

                # Assure that below methods produce the same results
                # For assertion set _init_files=True
                # assert (self.data_train.images == self.all_images[self.indices['train']]).all()
                # assert (self.data_train.masks  == self.all_masks[self.indices['train']]).all()
                # assert self.data_train.bboxes ==  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.data_train.masks ]
                # assert False, "all good"

                self.data_train.images = self.all_images[self.indices["train"]]
                # Setup for doing segmentation
                # self.data_train.masks = self.all_masks[self.indices['train']]
                # self.data_train.bboxes =  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.data_train.masks ]
                self.data_train.bboxes = self.all_bboxes[self.indices["train"]]
                self.data_train.indices = self.indices
                self.data_train._init_caches()

        # No augmentation here
        if stage == "fit" or stage == "validate" or stage == "val" or stage == None:
            if not hasattr(self, "data_val"):
                self.data_val = CropDataset(
                    self.data_dir,
                    dataset=self.dataset,
                    stage="val",
                    stack2_images=False,
                    image_size=self.image_size,
                    train_val_test_ratio=self.train_val_test_ratio,
                    seed=self.seed,
                    target_transform=self.test_transform,
                    half_precision=self.half_precision,
                    cache_bboxes=self.cache_bboxes,
                    cache_images=False,
                    _init_files=False,
                )

                # For assertion set _init_files=True
                # assert self.data_val.images == self.all_images[self.indices['val']]
                # assert self.data_val.masks  == self.all_masks[self.indices['val']]
                # assert self.data_val.bboxes ==  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.data_val.masks ]

                self.data_val.images = self.all_images[self.indices["val"]]
                # self.data_val.masks = self.all_masks[self.indices['val']]
                # self.data_val.bboxes =  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.data_val.masks ]
                self.data_val.bboxes = self.all_bboxes[self.indices["val"]]
                self.data_val.indices = self.indices
                self.data_val._init_caches()

        elif stage == "test":
            self.data_test = CropDataset(
                self.data_dir,
                dataset=self.dataset,
                stage="test",
                stack2_images=False,
                image_size=self.image_size,
                train_val_test_ratio=self.train_val_test_ratio,
                seed=self.seed,
                target_transform=self.test_transform,
                half_precision=self.half_precision,
                cache_bboxes=False,
                cache_images=False,
                _init_files=False,
            )

            # For assertion set _init_files=True
            # assert self.data_test.images == self.all_images[self.indices['test']]
            # assert self.data_test.masks  == self.all_masks[self.indices['test']]
            # assert self.data_test.bboxes ==  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.data_test.masks ]

            self.data_test.images = self.all_images[self.indices["test"]]
            # self.data_test.masks = self.all_masks[self.indices['test']]
            # self.data_test.bboxes =  [os.path.join(bbox_dir, os.path.basename(file)[:-4]+".csv") for file in self.data_test.masks ]
            self.data_test.bboxes = self.all_bboxes[self.indices["test"]]
            self.data_test.indices = self.indices

        elif stage == "predict":
            raise NotImplementedError("Setting stage to predict.")
        else:
            raise ValueError(
                f"Invalid stage '{stage}'",
            )

    # @wraps(pl.core.hooks.DataHooks.train_dataloader)
    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=CropDataset.bbox_collate,
            shuffle=True,
        )

    # @wraps(pl.core.hooks.DataHooks.val_dataloader)
    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=int(self.batch_size * self.eval_batch_scale),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=CropDataset.bbox_collate,
            shuffle=False,
        )

    # @wraps(pl.core.hooks.DataHooks.test_dataloader)
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=int(self.batch_size * self.eval_batch_scale),
            num_workers=self.num_workers,
            collate_fn=CropDataset.bbox_collate,
            shuffle=False,
        )

    def check_dataloaders(self):
        # Setup the variables
        if not hasattr(self, "data_train") or self.data_train is None:
            self.setup()
        if not hasattr(self, "data_test") or self.data_test is None:
            self.setup("test")
        try:
            # Change variables for testing
            real_batchsize = self.batch_size
            real_numworkers = self.num_workers
            self.num_workers = 0
            self.batch_size = 5
            return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
        finally:
            # reset variables
            self.batch_size = real_batchsize
            self.num_workers = real_numworkers

    # def predict_dataloader(self): # This will raise an error from super
    #    raise NotImplementedError("predict_dataloader")


class YOLO_PL(pl.LightningModule):
    # criterion = DetectionTrainer.criterion
    # postprocess = Yolov7Model.postprocessing

    def __init__(self, settings):
        super().__init__()
        # v8
        # self.yolo_model = YOLO(yolo_model, task="detect")
        # self.yolo_model.overrides = {} # default
        # self.yolo_model([*np.zeros((2, 1088, 1920))]) # init some stuff
        # self.model = self.yolo_model.model
        assert 0 <= settings["dataset"]["use_extra_class"] <= 1, "No or only one extra class can be used."
        assert settings["dataset"]["image_size"][0] == settings["dataset"]["image_size"][1], (
            "Image size must be a square."
        )

        self.settings = settings
        # self.example_input_array = torch.zeros((1, 3, *settings["dataset"]["image_size"]))
        # if settings["trainer"]["precision"] == 16 or settings["trainer"]["precision"] == "16-mixed":
        #    self.example_input_array = self.example_input_array.to(torch.float16)

        try:
            self.model = create_yolov7_model(
                settings["model"]["name"],
                num_classes=(
                    len(DATASETS[settings["dataset"]["name"]].get_label_ids()) + settings["dataset"]["use_extra_class"]
                ),
                pretrained=settings["model"]["pretrained"],
            )
        except AttributeError:
            # NOTE: There is sometimes an Attribute error that can be solved in repeating the process:
            self.model = create_yolov7_model(
                settings["model"]["name"],
                num_classes=(
                    len(DATASETS[settings["dataset"]["name"]].get_label_ids()) + settings["dataset"]["use_extra_class"]
                ),
                pretrained=settings["model"]["pretrained"],
            )

        if settings["model"].get("load_anchors", False):
            anchorparam = settings["model"]["load_anchors"]
            if anchorparam == True:  # auto
                try:
                    anchors = torch.load("./anchors/new_anchors_" + settings["model"]["name"] + ".pt").detach()
                    anchors.requires_grad = False
                    self.update_anchors(anchors)
                except Exception as e:
                    print(e, "\n^^^^^^^^^^^^\n", "Cannot update anchors")
            elif isinstance(anchorparam, str):
                self.update_anchors(torch.load(anchorparam).detach())  # path
            else:
                self.update_anchors(anchorparam.detach())  # pass manually
            print("Updated anchor settings")

        size = max(settings["dataset"]["image_size"])
        self.loss_func = create_yolov7_loss(self.model, image_size=size, **self.settings["loss"])
        self.image_size = (size, size)
        # Maybe adjust this because predifined turns into xyxy
        self.mAP = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            class_metrics=False,
            iou_thresholds=[settings["mAP"]["iou_threshold"]],
            rec_thresholds=settings["mAP"].get("rec_thresholds", None),
        )  # Single IoU threshold)
        self.test_mAP = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=True)

        self.save_hyperparameters()

    def forward(self, x):
        # print(x.dtype)
        # print(self.dtype)
        return self.model(x)

    @wraps(Yolov7Model.postprocess)
    def postprocess(self, *args, **kwargs):
        return self.model.postprocess(args, kwargs)

    def on_fit_start(self):
        self.loss_func.to(self.device)
        print(
            "Fitting",
            self.settings["model"]["name"],
            "on",
            self.settings["dataset"]["name"],
            "with",
            self.settings["optimizer"]["name"],
            end="\n\n",
        )

    def on_eval_start(self):
        self.loss_func.to(self.device)

    def on_test_start(self):
        self.loss_func.to(self.device)

    def on_train_epoch_start(self):
        self.loss_func.train()

    # For debugging
    def on_train_epoch_end(self):
        if self.trainer.current_epoch == 0:
            # log memory usage
            # for example for node training where this can not be called directly
            print(os.system("nvidia-smi"))

    def on_validation_epoch_start(self):
        self.loss_func.eval()

    def training_step(self, batch, _):
        images, labels, image_id, image_size = batch
        preds = self.model(images)
        loss, _ = self.loss_func(fpn_heads_outputs=preds, targets=labels, images=images)
        self.log("train_loss", loss, batch_size=images.shape[0], prog_bar=True)
        return loss

    def update_map(self, labels, preds, map):
        filtered_preds: list = self.model.postprocess(
            preds,
            conf_thres=self.settings["mAP"]["min_confidence"],  # TODO conf thresholds should be same?
            max_detections=self.settings["mAP"]["max_detections"],
            multiple_labels_per_box=False,
        )

        target_lists = [
            {"boxes": labels[(idx := labels[:, 0] == i), 2:], "labels": labels[idx, 1]}
            for i in range(len(filtered_preds))
        ]
        pred_lists = [None] * len(filtered_preds)
        for i, pred in enumerate(filtered_preds):  # len of batch_size
            pred = pred[pred[:, 4] > self.settings["mAP"]["nms_threshold"]].detach()

            nms_idx = torchvision.ops.batched_nms(
                boxes=pred[:, :4],
                scores=pred[:, 4],
                idxs=pred[:, 5],
                iou_threshold=self.settings["mAP"]["iou_threshold"],  # self.settings["mAP"]["iou_threshold"],
            )
            nms_preds = pred[nms_idx]
            boxes = torchvision.ops.box_convert(
                nms_preds[:, :4],
                "xyxy",
                "cxcywh",  # ughhh, is transformed in preprocess, maybe convert these others
            )
            boxes[:, [1, 3]] /= self.image_size[0]  # normalized height 0-1
            boxes[:, [0, 2]] /= self.image_size[1]  # normalized width 0-1
            pred_lists[i] = {"boxes": boxes, "scores": nms_preds[:, 4], "labels": nms_preds[:, 5]}
            # pred_list=[{"boxes"  : boxes,
            #  "scores" : nms_preds[:, 4],
            #  "labels" : nms_preds[:, 5]}]
            # map.update(pred_list, target_lists[i:i+1])
        map.update(pred_lists, target_lists)  # do not use forward else will compute & return

        # TODO: Plot bounding boxes with NMS supression.

    # Has the form
    """
    {'map': tensor(0.6000),
     'map_50': tensor(1.),
     'map_75': tensor(1.),
     'map_large': tensor(0.6000),
     'map_medium': tensor(-1.),
     'map_per_class': tensor(-1.),
     'map_small': tensor(-1.),
     'mar_1': tensor(0.6000),
     'mar_10': tensor(0.6000),
     'mar_100': tensor(0.6000),
     'mar_100_per_class': tensor(-1.),
     'mar_large': tensor(0.6000),
     'mar_medium': tensor(-1.),
     'mar_small': tensor(-1.)}
    """

    def _log_images(self, images, labels, image_id, predictions):
        # plot 10 bounding boxes with highest confidence
        # TODO. this relies on lost code and has not yet been restored
        figures = []
        IDX = slice(4)
        pred = self.model.postprocess(predictions, conf_thres=0.00, max_detections=30, multiple_labels_per_box=False)

        # TODO: Plot bounding boxes with NMS supression.
        # get the 10 bounding boxes with highest confidence
        # bboxes =pred[:, :4]
        # scores=pred[:, 4]
        # idxs  =pred[:, 5]
        # annotate with label and confidence
        # print("Images shape", images.shape)
        # imgs = np.ascontiguousarray(.permute(0, 2,3,1).cpu()) # old method

        bbox_files = [
            os.path.split(f)[1] for f in self.trainer.val_dataloaders.dataset.bboxes[image_id[IDX].cpu()]
        ]  # filenames to to true bounding boxes
        plots = utils.visualize_bboxes(
            DATASETS[self.settings["dataset"]["name"]],
            images=images[IDX].cpu(),
            bbox_files=bbox_files,
            predictions=[p.cpu() for p in pred[IDX]],
            plot_true_bboxes=self.current_epoch == 0,
            data_path=DATA_PATH,  # TODO this is globall!
            dataset=self.settings["dataset"]["name"],
            output_dir=None,
            font_scale=0.55,
            save_to_disk=False,
            crop_borders=240
            if self.settings["dataset"]["image_size"][0] > 1200
            else 100,  # certically to read original format
            target_width=640,
            pbar=False,
        )
        stack = torch.tensor(np.array(plots))  # turn into array first for speed up
        # Turn Batch x HxWxChannels into -> B x C x H x W
        grid = torchvision.utils.make_grid(
            stack.permute(0, 3, 1, 2),
            nrow=2,
        )
        # Turn into HxWxC
        self.logger.experiment.add_image("val_bboxes", grid.permute(1, 2, 0), self.current_epoch, dataformats="HWC")

    def validation_step(self, batch, batch_idx):
        images, labels, image_id, image_size = batch
        preds = self.model(images)
        loss, _ = self.loss_func(fpn_heads_outputs=preds, targets=labels, images=images)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(images), sync_dist=True)

        try:
            if batch_idx == 0:
                self._log_images(images, labels, image_id, preds)
        except Exception as e:
            print(e, "\n^^^^^^^^^^^^^^^^^^^^^^^^^^^\n Could not log images")
            raise
        self.update_map(labels, preds, self.mAP)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels, image_id, image_size = batch
        preds = self.model(images)
        loss, _ = self.loss_func(fpn_heads_outputs=preds, targets=labels, images=images)
        self.log("test_loss", loss, prog_bar=True, batch_size=len(images), sync_dist=True)
        self.update_map(labels, preds, self.test_mAP)

    # See: https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html
    def on_validation_epoch_end(self):
        # print("computing map-val", self.current_epoch)
        results = self.mAP.compute()
        # print("logging-val")
        if results["map"] != -1:
            self.log("hp_metric", results["map"], sync_dist=True)
            self.log("map", results.pop("map"), prog_bar=True, sync_dist=True)
        self.log_dict({k: v for k, v in results.items() if v != -1}, sync_dist=True)
        self.mAP.reset()

    def on_test_epoch_end(self):
        results = self.test_mAP.compute()
        map_per_class = results.pop("map_per_class")
        mar_per_class = results.pop("mar_100_per_class")
        if self.test_mAP.class_metrics:
            # Not one element tensors
            names = list(map(DATASETS[self.settings["dataset"]["name"]].get_label_name, self.test_mAP._get_classes()))
            for i, name in enumerate(names):
                if map_per_class[i] != -1:
                    self.log(f"test_map_{name}", map_per_class[i], sync_dist=True)
                if mar_per_class[i] != -1:
                    self.log(f"test_mar100_{name}", mar_per_class[i], sync_dist=True)
                print(f"test_map_{name}", map_per_class[i], f"\ntest_mar100_{name}", mar_per_class[i])
        else:
            print("not logging class metrics")
        self.log_dict({"test_" + key: val for key, val in results.items() if val != -1}, sync_dist=True)
        self.test_mAP.reset()

    # DEBUG
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #    print(batch.shape, batch.dtype)
    #    print("\n\ntranserfering")
    #    print(batch)
    #    #print(images.dtype, labels.dtype)
    #    return batch

    def update_anchors(self, anchors):
        try:
            update_func = self.model.update_anchors  # got renamed
        except AttributeError:
            update_func = self.model.update_model_anchors
        update_func(anchors)

    def configure_optimizers(self):
        steps_per_epoch = len(self.trainer.datamodule.data_train.images) // self.settings["dataset"]["batch_size"] + 1
        if self.settings["optimizer"]["weight_decay"] == "auto":
            self.settings["dataset"]["accumulate_batch_size"]
            num_accumulate_steps = max(
                round(self.settings["dataset"]["accumulate_batch_size"] / self.settings["dataset"]["batch_size"]), 1
            )
            base_weight_decay = 0.0005
            weight_decay = (
                base_weight_decay
                * self.settings["dataset"]["batch_size"]
                * num_accumulate_steps
                / self.settings["dataset"]["accumulate_batch_size"]
            )
            self.settings["optimizer"]["weight_decay"] = weight_decay
            print("Auto: weight decay set to", weight_decay)

        print("model type", self.dtype)

        param_groups = self.model.get_parameter_groups()

        if self.settings["optimizer"]["name"] == "SGD":
            optimizer = torch.optim.SGD(
                param_groups["other_params"],
                lr=self.settings["optimizer"]["learning_rate"],
                momentum=0.937,
                nesterov=True,
            )
            optimizer.add_param_group(
                {"params": param_groups["conv_weights"], "weight_decay": self.settings["optimizer"]["weight_decay"]}
            )
        elif self.settings["optimizer"]["name"] == "DeepSpeedCPUAdam":
            optimizer = DeepSpeedCPUAdam(
                param_groups["other_params"],
                lr=self.settings["optimizer"]["learning_rate"],
                betas=(0.937, 0.999),
                weight_decay=self.settings["optimizer"]["weight_decay"],
            )
            optimizer.add_param_group(
                {"params": param_groups["conv_weights"], "weight_decay": self.settings["optimizer"]["weight_decay"]}
            )
        elif self.settings["optimizer"]["name"] == "FusedAdam":
            optimizer = FusedAdam(
                param_groups["other_params"],
                lr=self.settings["optimizer"]["learning_rate"],
                betas=(0.937, 0.999),
                adam_w_mode=True,
            )
            optimizer.add_param_group(
                {"params": param_groups["conv_weights"], "weight_decay": self.settings["optimizer"]["weight_decay"]}
            )

        # """
        # optimizer = torch.optim.RAdam(self.parameters(),
        #                              lr=(self.settings["optimizer"]["learning_rate"]
        #                                  * ((1/self.settings["lr_scheduler"]["warmup_multiplier"]
        #                                      )** self.settings["lr_scheduler"]["warmup_epochs"])), # scale down for warmup
        #                              eps=1e-08,
        #                              betas= (0.937, 0.999),
        #                              weight_decay=self.settings["optimizer"]["weight_decay"])
        # """

        # lr_scheduler = CosineLrScheduler(optimizer,
        #    total_num_epochs=trainer.max_epochs,
        #    num_update_steps_per_epoch = num_batches,
        #    num_warmup_epochs=3,
        #    num_cooldown_epochs=3,
        #    k_decay=2,
        # )

        if self.settings["optimizer"].get("lr_scheduler", "").lower() == "onecyclelr":
            print("Using OneCycleLR scheduler")
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=self.settings["optimizer"]["learning_rate"],
                # NOTE: TODO: self.trainer.estimated_stepping_batches is better here but not compatible with the current caching setup
                # Probably a bug in lightning sets up the dataset twice :/
                # total_steps=self.trainer.estimated_stepping_batches,
                steps_per_epoch=steps_per_epoch,
                epochs=self.settings["trainer"]["max_epochs"],
                base_momentum=self.settings["lr_onecycle"]["base_momentum"],
                max_momentum=self.settings["lr_onecycle"]["max_momentum"],
                pct_start=self.settings["lr_onecycle"]["pct_start"],
                div_factor=self.settings["lr_onecycle"]["div_factor"],
                final_div_factor=self.settings["lr_onecycle"]["final_div_factor"],
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

        print("Using Multistep+CosineAnnealingLR scheduler")
        scheduler1 = MultiStepLR(
            optimizer,
            milestones=list(range(1, self.settings["lr_scheduler"]["warmup_epochs"] + 1)),
            gamma=self.settings["lr_scheduler"]["warmup_multiplier"],
        )  # warmup
        scheduler2 = MultiStepLR(
            optimizer,
            milestones=self.settings["lr_scheduler"]["milestones"],
            gamma=self.settings["lr_scheduler"]["milestone_multiplier"],
        )  # warmup
        scheduler3 = CosineAnnealingLR(optimizer, self.trainer.max_epochs, eta_min=0, last_epoch=-1)
        lr_scheduler = ChainedScheduler([scheduler1, scheduler2, scheduler3])
        return [optimizer], [lr_scheduler]
