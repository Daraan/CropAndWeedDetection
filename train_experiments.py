import sys
import os
import argparse
import time
import json
import warnings
import utils

from pprint import pprint

print("loading torch...", end="\r")
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
torch.set_float32_matmul_precision("medium")
print("imported torch", end="\r")

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GradientAccumulationScheduler, EarlyStopping
from pytorch_lightning.strategies import DeepSpeedStrategy
#from deepspeed.ops.adam import DeepSpeedCPUAdam

SEED = 4222
pl.seed_everything(SEED)

import albumentations as A
from albumentations.pytorch import ToTensorV2

print("imported lightning and albumentations. Importing model framework...")

print("Importing Frameworks")
from model import CNW_PATH, OPTUNA_DB_PATH, MODEL_PATH, DATA_PATH
from model import YOLO_PL
from model import CropAndWeedDataModule
from model import BBOX_PARAMS, DATASETS



# Path were the downloaded files images, segmentation masks, ... will be
if os.getcwd().startswith("/ceph") or os.getcwd().startswith("/pfs"): # dws or bw server 
    DATA_PATH = os.path.join(CNW_PATH, "data")
else:
    DATA_PATH =r"../data"
DATA_PATH = os.environ.get("DATA_PATH", DATA_PATH)

def parse_arguments():
    parser = argparse.ArgumentParser(description='setup training')
    parser.add_argument('--settings', type=str, default='./settings.json',
                        help='specify file with training specifications')
    parser.add_argument('--ckpt_path', required=False,
                    help='Optionally load a checkpoint.')
    parser.add_argument('--no_log', required=False, default=True,
                    help='Optionally load a checkpoint.')                

    args = parser.parse_args()
    return args

def create_trainer(settings, logging=True, logname=None):
    BATCH_SIZE = settings["dataset"]['batch_size']
    MAX_EPOCHS = settings['trainer']['max_epochs']
    IMAGE_SIZE = settings['dataset']['image_size']
    
    # disable logging for example when debugging
    if logging:
        filename = logname or settings["dataset"]["name"]+"_"+str(IMAGE_SIZE[0])+"px_"+settings["model"]["name"]+"_{epoch}_lr="\
                                 "_batch="+str(BATCH_SIZE)+"_{val_loss:.3f}_{map:.3f}"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            #=1,
            #dirpath=MODEL_PATH, # -> will be on logger path if None
            filename=(logname+"{map:.3f}") or filename,
            monitor="map",
            mode="max",
            verbose=1,
            save_last=False
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = filename+"-last"

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        tensor_logger  = pl.loggers.TensorBoardLogger("./lightning_logs/experiments", 
                                                      name= logname or settings["dataset"]["name"]+"_"+str(IMAGE_SIZE[0])+"px_"+settings['model']["name"]+f"max_epochs={MAX_EPOCHS}_fp={settings['trainer']['precision']}",
                                                      default_hp_metric=True,
                                                      version=0,
                                                      log_graph=False)

        callbacks = [checkpoint_callback]
    else:
        callbacks = []

    if settings["trainer"]["precision"] != 32 or settings["trainer"]["strategy"] == "deep": # assuming Deepspeed
        #strategy=DeepSpeedStrategy(config=deepspeed_config)
        deepspeed_config = {
                        #"optimizer" : "SGD", # SGD does not change
                        "zero_allow_untested_optimizer": True,
                        "train_batch_size" : settings['trainer']['deepspeed_config']['train_batch_size'],
                        "train_micro_batch_size_per_gpu" : BATCH_SIZE,
                        "zero_optimization": {
                                "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
                                "offload_optimizer": True,  # Enable Offloading optimizer state/calculation to the host CPU
                                "contiguous_gradients": True,  # Reduce gradient fragmentation. Usefull on larger models
                                #"overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed. When training across multiple GPUs/machines.
                                "allgather_bucket_size": 5e8,  # Number of elements to all gather at once.
                                "reduce_bucket_size": 5e8,  # Number of elements we reduce/allreduce at once.
                                "offload_optimizer": {
                                    "device": "cpu",
                                    "pin_memory": True,
                                },
                            },
                        "fp16": {
                            "enabled": True,
                            "min_loss_scale": 0.0001,
                            #"initial_scale_power" : 8, 
                            #"scale-tolerance" : 0.25,
                            #"fp16-scale-tolerance" : 0.25,
                            "hysteresis" : 4,
                        },
                        "scale-tolerance" : 0.25,
                        "fp16-scale-tolerance" : 0.25,
                        #"amp" : {
                        #    "enabled" : True,
                        #   "opt_level": "O1",
                        #},
                        "zero_force_ds_cpu_optimizer": False,
                        "load_full_weights" : True
                        }
        deepspeed_config.update(settings['trainer']['deepspeed_config'])

        strategy = DeepSpeedStrategy(config=deepspeed_config,  logging_batch_size_per_gpu=BATCH_SIZE, load_full_weights=True)
        #strategy = "auto"
        print("Using Deepseed", isinstance(strategy, DeepSpeedStrategy), "batch size", BATCH_SIZE, end="\n\n")
    else:
        strategy = settings["trainer"]["strategy"]
        if isinstance(settings["trainer"]["accumulate_grad_batches"], dict):
            #default {0: 8, 4: 4, 8: 1}
            settings["trainer"]["accumulate_grad_batches"] = {int(k):v for k,v in settings["trainer"]["accumulate_grad_batches"].items()}
            accumulator = GradientAccumulationScheduler(scheduling=settings["trainer"]["accumulate_grad_batches"])
            callbacks.append(accumulator)

    trainer = Trainer(  
        max_epochs=MAX_EPOCHS,
        #max_steps=7000,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else 1,  # limiting got iPython runs
        callbacks=callbacks,
        logger=[tensor_logger] if logging else [], # add fit_logger
        fast_dev_run=False, # DEBUG
        precision=settings['trainer']['precision'],
        strategy = strategy,
        #track_grad_norm=2, # older API
        #gradient_clip_val=0.5,
        accumulate_grad_batches=settings['trainer']["accumulate_grad_batches"] if strategy != "auto" else 1,
        num_sanity_val_steps=0,
        #val_check_interval=1, # set to 0 or false for default
        #check_val_every_n_epoch= None if settings['trainer'].get('val_check_interval', None) else settings['trainer'].get('check_val_every_n_epoch', 1)
    )
    print("\n")
    if not logging:
        print("NOTE: Logging of the model is disabled!")
    return trainer

def make_model(settings):
    """
    This creates a YOLOv7 model instance
    this function allows to load various preloaded weight types,
    even when the input layer or detection heads are incompatible.
    """
    
    # Setup / load model
    if settings['model'].get('weight_path', None):
        try:
            load_path = os.path.join(MODEL_PATH, settings['model']['weight_path'])
            try:
                pl_model = YOLO_PL.load_from_checkpoint(load_path, settings=settings)
            except FileNotFoundError: # check absolute path or local path
                load_path = settings['model']['weight_path']
                pl_model = YOLO_PL.load_from_checkpoint(load_path, settings=settings)
        except IsADirectoryError:  # ZeRO directory -> convert weights
            pl_model = YOLO_PL(settings)
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            load_state_dict_from_zero_checkpoint(pl_model, load_path, tag=None)
            #pl_model.load_state_dict(torch.load(path2))
        except RuntimeError as e:  # Assuming wrong projection head -> load all other weights
            pl_model = YOLO_PL(settings)
            # Create the 2nd model weights explicitly
            from copy import deepcopy
            settings2 = deepcopy(settings)
            
            # NOTE: todo: still assumes same input size
            settings2["dataset"]["name"] = settings2["model"]["weight_dataset"]
            settings2["model"]["pretrained"] = False
            model2 = YOLO_PL.load_from_checkpoint(load_path, settings=settings2)
            
            # Transfer the weights
            for c, c2 in zip(pl_model.model.model.children(), model2.model.model.children()):
                try:
                    c.load_state_dict(c2.state_dict())
                except RuntimeError as e:
                    print(e.args[0].split("\n")[0])
                    print("If above Error is for detection head then this is fine")
            print("Loaded weights expect the detection head.")
            del c, c2, settings2, model2
            
        print("loaded weights", settings['model']['weight_path'], end="\n\n")
    else:
        pl_model = YOLO_PL(settings)
        
    return pl_model



def make_augmentations(settings):
    IMAGE_SIZE = settings['dataset']['image_size']
    
    TRANSFORM_RESIZE = [A.LongestMaxSize(max(IMAGE_SIZE)),
                    A.PadIfNeeded(
                        IMAGE_SIZE[0], # height
                        IMAGE_SIZE[1], # width
                        border_mode=0,         # here we could mirror -> bad for bounding boxes, not cloned
                        value=(114/255, 114/255, 114/255), # color gray
                    ),]

    no_transform = A.Compose([*TRANSFORM_RESIZE, ToTensorV2()], bbox_params=BBOX_PARAMS)
    
    augmentation_pipeline = A.Compose([
        # Make size compatible with YOLO network
        # if STACK_2IMAGES is set this is ignored as images have correct size already from data loader
        *([A.LongestMaxSize(max(IMAGE_SIZE)),
          A.PadIfNeeded(IMAGE_SIZE[0], # height
                      IMAGE_SIZE[1], # width
                      border_mode=0,         # here we could mirror -> but bounding boxes are not clones

                       value=(114/255, 114/255, 114/255), # using mean of dataset ~roughly grey
                        )
          ] if not settings['dataset']['stack2_images'] else ()),
        # Transforms without increasing hardness of the task
        A.RandomRotate90(),
        A.Flip(p=0.5),
        # Color augmentations
        # grayscale or color
        # Recommended Jitter Augmentation parameters taken from:
        # https://github.com/WongKinYiu/yolov7/blob/711a16ba576319930ec59488c604f61afd532d5a/data/hyp.scratch.custom.yaml
        A.OneOf([A.ToGray(p=2), 
                 A.Sequential([A.ColorJitter(p=1.0, 
                                 brightness=(0.6, 1.4), 
                                 contrast=(0.8, 1.2), 
                                 saturation=(0.7, 1.3), 
                                 hue=(-0.015, 0.015)), 
                               #A.InvertImg(p=0.2)
                              ], 
                              p=8
                   )],
                p=0.75
        ),

        # Cropping 
        # As this zooms in on the picture this might make training easier
        # therefore keeping probability low
        # on the other hand aspect ratio gets a bit worse,
        A.RandomSizedBBoxSafeCrop(*IMAGE_SIZE, p=0.15),
        # Distortion
        # Normalize with imagenet weights
        *((A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=5),) if settings["dataset"]["normalize_images"] else ()), # normalizing is bad for YOLO
        # To Tensort
        ToTensorV2()
    ], bbox_params=BBOX_PARAMS)
    return no_transform, augmentation_pipeline


 
from copy import deepcopy 
 
def strict_on_validation_epoch_end(self):
    #print("computing map-val", self.current_epoch)
                                     
    for mAP, prefix in zip([self.mAP, False], ["", "val_"]):

        #print("logging-val")
        if mAP is False:
            if (self.current_epoch+1) % 5:
                continue
            print("Computing string map")
            strict_valmAP = MeanAveragePrecision(box_format="cxcywh", 
                             iou_type="bbox", 
                             class_metrics=True) 
                             
            strict_valmAP.detections = self.mAP.detections
            strict_valmAP.detection_labels = self.mAP.detection_labels
            strict_valmAP.detection_scores = self.mAP.detection_scores
            strict_valmAP.groundtruths = self.mAP.groundtruths
            strict_valmAP.groundtruth_labels = self.mAP.groundtruth_labels   
            mAP = strict_valmAP
        results = mAP.compute()
        
        map_per_class = results.pop('map_per_class')
        mar_per_class = results.pop('mar_100_per_class')
        
        self.log("hp_metric", results["map"], sync_dist=True)
        self.log(prefix+"map", results.pop("map"), prog_bar=True, sync_dist=True)
        if mAP.class_metrics:
            # Not one element tensors
            names = list(map(DATASETS[self.settings['dataset']['name']].get_label_name, self.mAP._get_classes()))
            for i, name in enumerate(names):
                if map_per_class[i] != -1:
                    self.log(prefix+f"map_{name}", map_per_class[i], sync_dist=True)
                if mar_per_class[i] != -1:
                    self.log(prefix+f"mar100_{name}", mar_per_class[i], sync_dist=True)
                print(prefix+f"map_{name}", map_per_class[i], f"\nval_mar100_{name}", mar_per_class[i])
        else:
            self.log_dict({prefix+k:v for k,v in results.items() if v != -1}, sync_dist=True)

    self.mAP.reset() 
    #strict_valmAP.reset()
 
def train(settings=None, logname=None):
    args = parse_arguments()
    pprint(args)
    if settings is None:
        with open(args.settings, "r") as f:
            settings = json.load(f)
    if args.ckpt_path:
        settings['trainer']['ckpt_path'] = args.ckpt_path
    
    # Validate settings or fix:
    if settings["trainer"]["precision"] != 32 or settings["trainer"]["strategy"] == "deep":
        if settings['trainer']['deepspeed_config']['train_batch_size'] == "batch_size":
            settings['trainer']['deepspeed_config']['train_batch_size'] = settings['dataset']['batch_size'] * settings['trainer'].get('accumulate_grad_batches', 1)
        elif settings['trainer']['deepspeed_config']['train_batch_size'] != settings['dataset']['batch_size']:
            raise ValueError("deepspeed and dataset batch_sizes do not match")
        if settings['trainer']['deepspeed_config']["train_micro_batch_size_per_gpu"] == "batch_size":
            # auto set
            settings['trainer']['deepspeed_config']["train_micro_batch_size_per_gpu"] = settings['dataset']['batch_size']
    
    
    if settings['model']['name'] == 'yolov7-tiny' and settings['model']['pretrained']:
        settings['model']['pretrained'] = False
        print("YOLOv7-tiny currently has no pretrained weights, disabling")
    
    # Paths
    cache_images = bool(int(os.environ.get("CACHE_IMAGES", "0")))
    cache_bboxes = bool(int(os.environ.get("CACHE_BBOXES", True)))
    BBOX_PATH = os.environ.get("BBOX_ROOT")
    IMAGES_PATH = os.environ.get("IMAGES_ROOT")
    
    if BBOX_PATH or IMAGES_PATH:
        data_path = DATA_PATH
        paths = {"bboxes" : BBOX_PATH or data_path,
                 "images" : IMAGES_PATH or data_path}
        data_path = paths
        print("Using customized paths:", data_path)
    else:
        data_path = DATA_PATH
        
    # Setups
    pprint(settings)
        
        
    #settings["trainer"]["max_epochs"] = 20
    trainer = create_trainer(settings, logging=args.no_log, logname=logname)
    
    no_transform, augmentation_pipeline = make_augmentations(settings)
    data = CropAndWeedDataModule(settings['dataset']['name'], data_path, 
                                      batch_size=settings['dataset']['batch_size'],
                                      num_workers=utils.check_worker_count(settings["dataset"]['num_workers']), 
                                      stack2_images=settings["dataset"]['stack2_images'],
                                      image_size=settings['dataset']['image_size'],
                                      train_transform=augmentation_pipeline,
                                      test_transform=no_transform,
                                      seed=settings['dataset']['seed'],
                                      cache_bboxes=cache_bboxes,
                                      cache_images=cache_images,
                                      #train_val_test_ratio=[0.15, 0.15, 0.7],
                                      half_precision=False)
    # Early load for testing
    #data.setup()
    print("...Data Module initialized")
    
    pl_model = make_model(settings)
    pl_model.to(torch.float32)    
    pl_model.model.to(torch.float32)
    
    print("Model created", end="\n\n")
    if settings['trainer']['ckpt_path']:
        try:
            trainer.fit(pl_model, data, ckpt_path=settings['trainer']['ckpt_path'])
        except Exception as e:
            raise
            print(e, "\n", "Training without loading checkpoint")
            trainer.fit(pl_model, data)
    else:
        trainer.fit(pl_model, data)
    
    print("\n\nTraining done")  

    if trainer.callback_metrics.get("map", 0) > 0.1:
        trainer.test(pl_model, data)
    return
    
    pl_model.to("cpu")
    pl_model.mAP = MeanAveragePrecision(box_format="cxcywh", 
                                     iou_type="bbox", 
                                     class_metrics=True)
                                     
    pl_model.strict_valmAP = MeanAveragePrecision(box_format="cxcywh", 
                                 iou_type="bbox", 
                                 class_metrics=True)                                 
    # swap out the function

    try:
        trainer.validate(pl_model.cuda(), data)
    finally:
        YOLO_PL.on_validation_epoch_end = standard_eval
    
    
    
def _experiments_present(file):
    for file in os.listdir("experiments"):
        print("Checking", file)
        if os.path.exists(os.path.join(".","lightning_logs", "experiments", os.path.splitext(file)[0])):
            print("path exists", os.path.join("lightning_logs", "experiments", file))
            return True
        for dir in os.listdir(os.path.join(".","lightning_logs", "experiments")):
            if not dir.startswith("settings"):
                pass
                

if __name__ == "__main__":
    standard_eval = YOLO_PL.on_validation_epoch_end
    YOLO_PL.on_validation_epoch_end = strict_on_validation_epoch_end
    for file in os.listdir("experiments"):
        print("Checking", file)
        if os.path.exists(os.path.join(".","lightning_logs", "experiments", os.path.splitext(file)[0])):
            print("path exists", os.path.join("lightning_logs", "experiments", file))
            continue
        if os.path.exists(os.path.join(".","lightning_logs", "experiments", "size", os.path.splitext(file)[0])):
            continue

        print(os.path.join("lightning_logs", "experiments", os.path.splitext(file)[0]), "does not exist")
        
        with open(os.path.join("experiments", file), "r") as f:
            settings = json.load(f)
        train(settings, logname=os.path.splitext(file)[0])

