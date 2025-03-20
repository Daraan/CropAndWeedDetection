import os
import warnings

# TODO: acquire these from a trial_settings.json
from collections import namedtuple
from copy import deepcopy
from typing import Union

import optuna
import torch
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.integration import PyTorchLightningPruningCallback as PruningCallback
from optuna.pruners import MedianPruner, PatientPruner, PercentilePruner
from optuna.samplers import PartialFixedSampler
from optuna.trial import FrozenTrial, TrialState
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import train_experiments
from model import *
from train import *

print("Optuna Version", optuna.__version__)


# NOTE: The builtin Callback is currently not compatible with pytorch-lightning
class PyTorchLightningPruningCallback(Callback):
    """
    # Code taken from: https://github.com/optuna/optuna-examples/issues/166#issuecomment-1403112861

    PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message, stacklevel=2)
            return

        self._trial.report(current_score.item(), step=epoch)
        print(format(epoch, ">3"), " : Validation mAP:", format(current_score, ".3f"))
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


HParamInt = namedtuple("HParamInt", "name, low, high, log, step", defaults={"log": False, "step": 1}.values())
HParamFloat = namedtuple("HParamFloat", "name, low, high, log, step", defaults={"log": False, "step": None}.values())


HP_SPACE = {  # "learning_rate" : (1e-4, 5e-2),
    # "warmup_epochs" : (0, 2),
    # "warmup_multiplier" : [0.15, 5],
    # OneCycleLr parameters
    # see: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
    "learning_rate": HParamFloat("learning_rate", 1e-3, 9e-2, log=True),
    "pct_start": HParamFloat("pct_start", 0.05, 0.4, step=0.01),  # length of warmup interval in terms of total steps
    "div_factor": HParamInt("div_factor", 10, 100, step=5),  # scaline
    "final_div_factor": HParamInt("final_div_factor", 100, 100_000, log=True),
    #'base_momentum' : HParamFloat(0.75, 0.85),
}

PRESETS = {
    "onecycle": {
        "default_parameters": {
            "learning_rate": 1e-2,
            "div_factor": 25.0,
            "final_div_factor": 10_000.0,
            "pct_start": 0.3,  # 3 epochs but on 50 total
        },
        "recommended_parameters": [
            # see: https://github.com/WongKinYiu/yolov7/tree/711a16ba576319930ec59488c604f61afd532d5a/data
            {
                "learning_rate": 1e-2,
                "div_factor": 10.0,
                "final_div_factor": 100,  # final learning rate 1e-2 / 100 = 1e-4
                "pct_start": 0.3,  # 3 epochs but on 50 total
            },
            # From experience
            {
                "learning_rate": 2.5e-2,
                "div_factor": 5.0,
                "final_div_factor": 500,  # final learning rate 1e-2 / 100 = 1e-4
                "pct_start": 0.25,  # 3 epochs but on 50 total
            },
        ],
        "edgecase_trials": [
            # Min LR, short warmup
            {
                "learning_rate": HP_SPACE["learning_rate"].low,
                "div_factor": HP_SPACE["div_factor"].low,  # using low division only, high division should be too low
                "pct_start": HP_SPACE["pct_start"].low,
                "final_div_factor": 10_000,  # default
            },
            # MinLR, long warmup                                # very bad
            {
                "learning_rate": HP_SPACE["learning_rate"].low,
                "div_factor": HP_SPACE["div_factor"].high,  # using low division only, high division should be too low
                "pct_start": HP_SPACE["pct_start"].high,
                "final_div_factor": 10_000,  # default
            },
            # Max LR, short high warmup                         # below average
            {
                "learning_rate": HP_SPACE["learning_rate"].high,
                "div_factor": HP_SPACE["div_factor"].low,
                "pct_start": HP_SPACE["pct_start"].low,
                "final_div_factor": 10_000,  # default
            },
            # Max LR, short low warmup
            {
                "learning_rate": HP_SPACE["learning_rate"].high,
                "div_factor": HP_SPACE["div_factor"].high,
                "pct_start": HP_SPACE["pct_start"].low,
                "final_div_factor": 10_000,  # default
            },
            # Max LR, long low warmup, this might work
            {
                "learning_rate": HP_SPACE["learning_rate"].high,
                "div_factor": HP_SPACE["div_factor"].high,
                "pct_start": HP_SPACE["pct_start"].high,
                "final_div_factor": 10_000,  # default
            },
            # Max LR, long high warmup, this will likely overflow and fail
            {
                "learning_rate": HP_SPACE["learning_rate"].high,
                "div_factor": HP_SPACE["div_factor"].low,
                "pct_start": HP_SPACE["pct_start"].high,
                "final_div_factor": 10_000,  # default
            },
        ],
    }
}


def suggest_hparam(trial, param: Union[HParamInt, HParamFloat]):
    """Automatic wrapper to suggest Int or float parameters"""
    if isinstance(param, HParamInt):
        return trial.suggest_int(param.name, low=param.low, high=param.high, step=param.step, log=param.log)
    elif isinstance(param, HParamFloat):
        return trial.suggest_float(param.name, low=param.low, high=param.high, step=param.step, log=param.log)
    raise ValueError("Not an Int or Float parameter")


def optimize():
    # get settings
    args = parse_arguments()
    pprint(args)
    with open(args.settings, "r") as f:
        settings = json.load(f)

    if settings["model"]["name"] == "yolov7-tiny" and settings["model"]["pretrained"]:
        print("yolov7-tiny cannot be pretrained. Setting pretrained=False")
        settings["model"]["pretrained"] = False

    trial_trainer_settings = deepcopy(settings["trainer"])
    TRIAL_EPOCHS = settings["trainer"]["max_epochs"]
    BATCH_SIZE = settings["dataset"]["batch_size"]
    IMAGE_SIZE = settings["dataset"]["image_size"]

    trial_trainer_settings["max_epochs"] = TRIAL_EPOCHS
    if TRIAL_EPOCHS > 20:
        print("WARNING: Trial Epochs are > 20, is this a mistake?")

    study_settings = deepcopy(settings)  # better overview this way

    fixed_params = {
        "batch_size": settings["dataset"]["batch_size"],
        "base_momentum": settings["lr_onecycle"]["base_momentum"],
        "max_momentum": settings["lr_onecycle"]["max_momentum"],
    }

    ####

    # Setup trial

    os.makedirs(os.path.split(OPTUNA_DB_PATH)[0], exist_ok=True)  # imported

    median_pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3, n_min_trials=3)
    # Keep 60% of trials instead of 50
    percentile_pruner = PercentilePruner(60.0, n_startup_trials=5, n_warmup_steps=3, n_min_trials=4)

    # Using patience because map can flucutate a bit over epochs
    # prevent pruning of trials that look good at thes start but make one mistake
    patient_percentile = PatientPruner(percentile_pruner, patience=2, min_delta=0.02)  # at least increase by 2%

    storage_url = "sqlite:///" + OPTUNA_DB_PATH
    study_name = (
        "cyclelr_"
        + settings["model"]["name"]
        + "_"
        + settings["dataset"]["name"]
        + "_"
        + str(settings["dataset"]["image_size"][0])
        + "px"
    )

    # DELETE STUDY, be careful when uncommenting
    if os.environ.get("DELETE_STUDY"):
        print("Deleting study", study_name)
        optuna.delete_study(study_name=study_name, storage=storage_url)

    study = optuna.create_study(
        study_name=study_name, storage=storage_url, direction="maximize", pruner=percentile_pruner, load_if_exists=True
    )

    study.set_user_attr("dataset", settings["dataset"]["name"])
    study.set_user_attr("model", settings["model"]["name"])

    # Adding non edgecases which likely could result in good performances
    # edgcase trials could be very bad, should be tested after n_startup_trials
    # so pruning them can happen

    if PRESETS["onecycle"]["default_parameters"]:
        study.enqueue_trial(PRESETS["onecycle"]["default_parameters"], skip_if_exists=True)
    if PRESETS["onecycle"]["recommended_parameters"]:
        for sample in PRESETS["onecycle"]["recommended_parameters"]:
            study.enqueue_trial(sample, skip_if_exists=True)

    def objective(trial):
        torch.cuda.empty_cache()
        # opt_type = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RAdam"])

        trial_settings = deepcopy(study_settings)
        # trial_settings["model"]["pretrained"] = trial.suggest_bool("pretrained") #using tiny # pretrained is always better
        batch_size = trial.suggest_int("batch_size", BATCH_SIZE, BATCH_SIZE)  # fixing this

        # Hyperparameters
        trial_settings["optimizer"]["learning_rate"] = suggest_hparam(trial, HP_SPACE["learning_rate"])
        trial_settings["lr_onecycle"]["div_factor"] = suggest_hparam(trial, HP_SPACE["div_factor"])
        trial_settings["lr_onecycle"]["final_div_factor"] = suggest_hparam(trial, HP_SPACE["final_div_factor"])
        trial_settings["lr_onecycle"]["pct_start"] = suggest_hparam(trial, HP_SPACE["pct_start"])

        # Constants
        trial_settings["trainer"]["max_epochs"] = TRIAL_EPOCHS

        trial.set_user_attr("model", trial_settings["model"]["name"])
        trial.set_user_attr("pretrained", trial_settings["model"]["pretrained"])
        trial.set_user_attr("dataset", trial_settings["dataset"]["name"])

        trial.set_user_attr("optimizer", trial_settings["optimizer"]["name"])
        trial.set_user_attr("image_size", IMAGE_SIZE)
        trial.set_user_attr("batch_size", BATCH_SIZE)
        trial.set_user_attr("max_epochs", TRIAL_EPOCHS)  # important for cosine annealing

        ######## Callbacks

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        tensor_logger = TensorBoardLogger(
            "./lightning_logs/" + trial_settings["dataset"]["name"],
            name=f"Trial-{trial_settings['model']['name']}_" + str(IMAGE_SIZE[0]) + "px_"
            f"lr={trial_settings['optimizer']['learning_rate']:.2g}"
            # f"warmup=({trial_settings['lr_scheduler']['warmup_epochs']},{trial_settings['lr_scheduler']['warmup_multiplier']:.2g})",
            f"cycle({trial_settings['lr_onecycle']['div_factor']}, {trial_settings['lr_onecycle']['final_div_factor']}, {trial_settings['lr_onecycle']['div_factor']}, {trial_settings['lr_onecycle']['pct_start']:.2f})",
            default_hp_metric=True,
        )

        # fit_csv_logger = pl.loggers.CSVLogger("./lightning_logs", name='dev_fit', version=None, prefix='', flush_logs_every_n_steps=100)

        trainer = Trainer(
            max_epochs=trial_settings["trainer"]["max_epochs"],
            # max_steps=7000,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else "auto",
            callbacks=[lr_monitor, PyTorchLightningPruningCallback(trial, monitor="map")],
            logger=[tensor_logger],  # add fit_logger
            fast_dev_run=False,
            num_sanity_val_steps=0,
            strategy=trial_trainer_settings["strategy"],
        )

        no_transform, augmentation_pipeline = train_experiments.make_augmentations(settings)
        data = CropAndWeedDataModule(
            trial_settings["dataset"]["name"],
            DATA_PATH,
            batch_size=batch_size,
            image_size=trial_settings["dataset"]["image_size"],
            num_workers=trial_settings["dataset"]["num_workers"],
            train_transform=augmentation_pipeline,
            test_transform=no_transform,
            stack2_images=True,
        )

        print("lr", trial_settings["optimizer"]["learning_rate"], trial_settings["lr_onecycle"]["div_factor"])
        model = YOLO_PL(trial_settings)
        try:
            trainer.fit(model, data)
        except optuna.exceptions.TrialPruned as e:
            trial.set_user_attr("end_epoch", trainer.current_epoch)
            raise e
        except Exception as e:
            print(e)
            raise

        return trainer.callback_metrics["map"].item()


if __name__ == "__main__":
    optimize()
