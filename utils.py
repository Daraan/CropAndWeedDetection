import os
import warnings
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from typing import Union, Literal, List
import cv2
import csv

import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 1
font_line_type = cv2.LINE_AA


def train_val_test_split(all_images, seed, train_val_test_ratio):
    assert sum(train_val_test_ratio) == 1, "ratios do not sum to 1"
    geny = np.random.default_rng(seed)
    shuffled = geny.choice(
        len(all_images), len(all_images), replace=False
    )  # not really necessary but from an older implementation
    train_val, test = train_test_split(shuffled, test_size=train_val_test_ratio[-1], random_state=seed)
    train, val = train_test_split(
        train_val,
        train_size=(train_val_test_ratio[0] / (train_val_test_ratio[0] + train_val_test_ratio[1])),
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


TEXT_COLOR = (40, 40, 40)


def visualize_bbox(*args, **kwargs):
    # code lost
    pass


def plot_masked_image(*args, **kwargs):
    # Code lost
    pass


def extract_statistics(dataset):
    """Extract mean and standard deviations for each color channel from an iterable dataset"""
    # This is saldy a code that also was lost
    # what it did was calculating a running mean and running variance for batch inputs.
    # The formula that was used originates from here:
    # https://math.stackexchange.com/a/2971563
    import torch

    return torch.tensor([0.5030, 0.4881, 0.4286]), torch.tensor([0.1899, 0.1851, 0.1753])


def check_worker_count(requested_workers) -> int:
    """
    Checks requested vs suggested worker count
    Code extracted from : https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader
    """
    max_num_worker_suggest = None
    cpuset_checked = False
    if hasattr(os, "sched_getaffinity"):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
            cpuset_checked = True
        except Exception:
            pass
    if max_num_worker_suggest is None:
        # os.cpu_count() could return Optional[int]
        # get cpu count first and check None in order to satify mypy check
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count

    if max_num_worker_suggest is None:
        warnings.warn(
            "DataLoader is not able to compute a suggested max number of worker in current system.", stacklevel=2
        )
        return requested_workers

    if requested_workers > max_num_worker_suggest:
        warnings.warn(
            (
                "This DataLoader's requested amount of {} worker processes is larger than the suggested amount of {} "
                "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary."
            ).format(requested_workers, max_num_worker_suggest),
            stacklevel=2,
        )
        return max_num_worker_suggest
    return requested_workers


def plot_predictions(model, dataloader, DATASET, max_detections=2, multiple_labels_per_box=False):
    """
    Note:
        this function is here only to demonstrate how the below function works
        but cannot be used from within utils!
    """
    images, labels, image_id, image_size = next(iter(dataloader))
    bbox_files = dataloader.dataset.bboxes[image_id]
    with torch.no_grad():
        preds: list = model(images)
    preds = model.model.postprocess(
        preds, conf_thres=0.00, max_detections=max_detections, multiple_labels_per_box=multiple_labels_per_box
    )
    cnw_dataset = DATASETS[DATASET]
    path1 = os.path.join(DATA_PATH, "visualization")
    path2 = os.path.join(path1, DATASET)
    path3 = os.path.join(path2, model.settings["model"]["name"])
    for path in [path1, path2, path3]:
        os.makedirs(path, exist_ok=True)
    utils.visualize_bboxes(
        cnw_dataset,
        images,
        bbox_files,
        predictions=preds,
        plot_true_bboxes=True,
        data_path=DATA_PATH,
        dataset=DATASET,
        output_dir=path3,
        target_width=1280,
    )


def make_label_box(
    image, bbox, text, color=(255, 0, 0), thickness=2, font_scale=0.5, font=cv2.FONT_HERSHEY_SIMPLEX, font_thickness=2
):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(text, font, font_scale, font_thickness)
    # rect_color = (20, 240, 30) if label_id == true_label else (220, 0, 80)
    if text:
        rect_color = color
        cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), rect_color, -1)
        cv2.putText(
            image,
            text=text,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=font,
            fontScale=font_scale,
            color=(0, 0, 0),
            lineType=font_line_type,
            thickness=font_thickness,
        )


def visualize_original_bboxes(
    cnw_dataset,
    dataset: str,
    data_path: Union[str, os.PathLike] = "./data",
    *,
    bbox_files: List[str] = None,
    save_to_disk: bool = True,
    output_dir: Union[str, os.PathLike] = "./data/visualization",
    image_filter: str = None,
    **kwargs,
) -> list:
    """
    Annotes the original images with the true bounding boxes.


    Parameters
    ----------
    cnw_dataset : cnw.datasets.dataset
        A cnw.datasets.dataset instance that hold information about names and
        color mappings.
    dataset : str
        Name of the dataset, must be the same as for the cnw_dataset but this
        does not have that information.
    data_path : Union[str, os.PathLike], optional
        Root folder which holds the images and bounding boxes.
        The default is "./data".

    ---
    Keyword only arguments:

    bbox_files : List[str], optional
        A list of file names for which the bounding bloxes are to be plotted.
        The default is None.
    save_to_disk : bool, optional
        Save the generated images to disk.
        The default is True.
    output_dir : Union[str, os.PathLike], optional
        If save_to_disk is True, where to store them.
        The default is './data/visualization'.
    image_filter : str, optional
        if bbox_files is None chooses files including this filter in a
            image_filter in filename
        check
        The default is None.
    target_width : int, optional
        Width of the resulting image.
        The default is 1280 pixels.
    font_scale : float, optional
        Scaling of annotation font to be used with cv2.
        The default is 0.8.
    pbar : bool, optional.
        Use a tqdm iterator as progress bar
        The default is True

    Returns
    -------
    list
        List of plotted image files.

    """
    return np.array(
        visualize_bboxes(
            cnw_dataset,
            images=None,
            data_path=data_path,
            plot_true_bboxes=True,
            bbox_files=bbox_files,
            predictions=None,
            dataset=dataset,
            output_dir=output_dir,
            crop_borders=0,
            image_filter=image_filter,
            save_to_disk=save_to_disk,
            **kwargs,
        )
    )  # [:,:,:,[2,1,0]]


def image_to_bbox_file(imfile, bbox_path, dataset="CropAndWeed"):
    name = os.path.split(imfile)[1]
    return os.path.join(bbox_path, dataset, os.path.splitext(name)[0] + ".csv")


DEBUG = False


def visualize_bboxes(
    cnw_dataset,
    images: list,
    bbox_files: list,
    predictions: list = None,  # plot true bounding boxes or pass them
    *,
    plot_true_bboxes=False,  # Not compatible with augmentation!
    data_path="./data",
    dataset="CropAndWeed",
    output_dir="./data/visualization",
    crop_borders=0,
    font_scale=0.8,
    target_width=1280,
    image_filter: str = None,  # has only and effect if bboxes=true,
    save_to_disk=True,
    mask_root: bool = None,
    pbar=True,  # tqdm progress bar
):
    """
    Visualize bounding boxes
    this function is inspired by and partially uses lines from:
    https://github.com/cropandweed/cropandweed-dataset/blob/main/cnw/visualize_annotations.py
    and
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    """

    # if dataset not in datasets.DATASETS:
    #    raise RuntimeError(f'dataset {dataset} not defined in datasets.py')
    if bbox_files is None or len(bbox_files) == 1:
        bbox_files = [bbox_files]
    images_dir = os.path.join(data_path, "images")
    # print(data_path, images_dir)
    train_bbox_dir = os.path.join(data_path, "bboxes")
    bboxes_dir = os.path.join(train_bbox_dir, f"{dataset}Eval")

    background_id = len(cnw_dataset.get_label_ids())

    written_images = []

    # print("bbox_dir", bboxes_dir)

    if mask_root:
        label_ids_dir = os.path.join(mask_root, dataset)
    if save_to_disk:
        visualizations_dir = os.path.join(output_dir, dataset)
        os.makedirs(visualizations_dir, exist_ok=True)
    if predictions is None and bbox_files[0] is None:
        DEBUG and print("1st if")
        iterator = tqdm(sorted(os.listdir(bboxes_dir))) if pbar else sorted(os.listdir(bboxes_dir))
        assert plot_true_bboxes, "Not bboxes passed and no ground truths"
        predictions = [None] * len(iterator)
    else:
        DEBUG and print("2nd if")

        def splitter(file):
            # print("file is", file)
            return os.path.split(file)[1]

            if "Eval" not in file:  # TODO: This could lead to buggs
                path, name = os.path.split(file)
                r = os.path.join(path + "Eval", name)
            print("changes file is", f)
            return r

        iterator = map(splitter, bbox_files)
        if pbar:
            iterator = tqdm(iterator, total=len(bbox_files))
        else:
            iterator = list(iterator)  # want len
    if predictions is None:
        predictions = [None] * len(iterator)
    if images is None:
        images = [None] * len(iterator)
    assert len(images) == len(iterator) == len(predictions), "Lenghts do not match"
    if len(images) == 0:
        warnings.warn("No images to plot - iterator empty")
    for bboxes_file, image, pred in zip(iterator, images, predictions):
        DEBUG and print(bboxes_file)
        target = os.path.split(bboxes_file)[1]  # filename.csv
        target = os.path.splitext(target)[0]  # rootname
        if image_filter is not None and image_filter not in target:
            continue
        # if os.path.exists(os.path.join(bboxes, dataset, bboxes_file)) and image_filter in target:
        label_count = defaultdict(int)
        image_path = os.path.join(images_dir, f"{target}.jpg")
        if image is None:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)  # NOTE: todo, will not match with augmented predictions!
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(image_path, "image not found")
        else:
            if image.max() <= 1:  # scale up as we work in int format.
                image *= 255
                image = np.ascontiguousarray(image.permute(1, 2, 0).cpu(), dtype=np.uint8)
            if not isinstance(image, np.ndarray):
                image = np.ascontiguousarray(image.permute(1, 2, 0).cpu(), dtype=np.uint8)
        # if isinstance(bboxes, str) and bboxes.lower() in ("true", "truth"):
        with open(os.path.join(bboxes_dir, bboxes_file), "r") as csv_file:
            csv_reader = csv.DictReader(
                csv_file, delimiter=",", fieldnames=["left", "top", "right", "bottom", "label_id", "stem_x", "stem_y"]
            )

            # true_labels = []

            if plot_true_bboxes:  # Not compatibe with ugmentations! Fixed
                if os.path.exists(image_path):
                    trueimage = cv2.imread(image_path)  # NOTE: todo, will not match
                    if pred is None:
                        image = trueimage
                    else:
                        trueimage = cv2.cvtColor(trueimage, cv2.COLOR_BGR2RGB)
                else:
                    raise FileNotFoundError(image_path)

            for row in csv_reader:
                label_id = int(row["label_id"])
                color = (0, 0, 0) if label_id == background_id else cnw_dataset.get_label_color(label_id, bgr=True)
                label_count[label_id] += 1
                # true_labels.append(label_id)
                if plot_true_bboxes:
                    # cv2.rectangle(trueimage, (int(row['left']), int(row['top'])),
                    #              (int(row['right']), int(row['bottom'])), color, thickness=3)
                    # cv2.circle(image, (int(row['stem_x']), int(row['stem_y'])), 15, color, thickness=2)
                    make_label_box(
                        trueimage,
                        (int(row["left"]), int(row["top"]), int(row["right"]), int(row["bottom"])),
                        text=cnw_dataset.get_label_name(label_id),
                        color=color,
                        thickness=5,
                        font_scale=font_scale * 3,
                        font_thickness=FONT_THICKNESS + 1,
                    )

        if pred is not None:
            # pascal voc format!
            boxes = pred[:, :4].cpu().numpy().astype(int)
            scores = pred[:, 4].cpu().numpy()
            label_ids = pred[:, 5].cpu().numpy().astype(int)

            for box, score, label_id in zip(boxes, scores, label_ids):
                score, label_id = score.item(), label_id.item()
                color = (0, 0, 0) if label_id == background_id else cnw_dataset.get_label_color(label_id, bgr=True)

                # BBox
                # x_min, y_min, x_max, y_max = box
                # Small label box
                name = cnw_dataset.get_label_name(label_id)
                name += f" {score:.1%}"
                make_label_box(
                    image, box, name, thickness=2, color=color, font_scale=font_scale, font_thickness=FONT_THICKNESS
                )

        if crop_borders:
            image = image[crop_borders // 2 : -crop_borders // 2]
        # Resize image
        target_size = (int(target_width * 1), int(image.shape[0] * (target_width * 1 / image.shape[1])))
        image = cv2.resize(image, target_size, cv2.INTER_LINEAR)

        # Write image path
        cv2.putText(
            image,
            target,
            (10, 25),
            font,
            font_scale * 1.3,
            (255, 255, 255),
            thickness=FONT_THICKNESS,
            lineType=font_line_type,
        )
        if plot_true_bboxes and pred is not None:
            trueimage = cv2.resize(trueimage, target_size, cv2.INTER_LINEAR)
            image = cv2.hconcat([image, trueimage])

        # For segmentation
        if mask_root:
            # TODO
            raise NotImplementedError("Part needs some update")
            label_ids_path = os.path.join(label_ids_dir, f"{target}.png")

            label_layer = (
                cv2.resize(ids2colors(cv2.imread(label_ids_path, 0), cnw_dataset), target_size, cv2.INTER_LINEAR)
                if os.path.exists(label_ids_path)
                else np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            )
            image = cv2.vconcat([image, label_layer])

        ((_, text_height), _) = cv2.getTextSize("dummy", font, font_scale * 1.25, FONT_THICKNESS + 1)
        image = cv2.copyMakeBorder(image, 0, text_height + 18, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        offset = 5
        for label_id, count in sorted(label_count.items(), key=lambda item: item[1], reverse=True):
            label_name = cnw_dataset.get_label_name(label_id)
            if label_name is not None:
                text = f"{'Vegetation' if label_name is None else label_name} ({count})  "
                image = cv2.putText(
                    image,
                    text,
                    (offset, target_size[1] + text_height + 3),
                    font,
                    font_scale * 1.25,
                    color=cnw_dataset.get_label_color(label_id, bgr=True),
                    thickness=FONT_THICKNESS + 1,
                    lineType=font_line_type,
                )
                offset += cv2.getTextSize(text, font, font_scale * 1.3, FONT_THICKNESS + 1)[0][0] + 1
        if pred is None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if save_to_disk:
            cv2.imwrite(os.path.join(visualizations_dir, f"{target}.jpg"), image)
        # written_images.append(os.path.join(visualizations_dir, f'{target}.jpg'))
        written_images.append(image)
        # return written_images
    if pbar:
        iterator.close()
    return written_images
