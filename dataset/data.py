from ast import literal_eval
import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd
from datasets import Dataset


# PERCENTAGE_OF_TRAIN = 1.0
# # PERCENTAGE_OF_VAL_SET_TO_USE = 0.2
# PERCENTAGE_OF_VAL= 1.0



def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302
    IMAGE_INPUT_SIZE=512

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_df(path_full_dataset):
    PERCENTAGE_OF_TRAIN = 1.0
    PERCENTAGE_OF_VAL= 1.0
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]

    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

    datasets_as_df = {dataset: os.path.join(path_full_dataset, dataset) + ".csv" for dataset in ["train", "valid"]}
    datasets_as_df = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_df.items()}

    total_num_samples_train = len(datasets_as_df["train"])
    total_num_samples_val = len(datasets_as_df["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL * total_num_samples_val)

    # log.info(f"Train: {new_num_samples_train} images")
    # log.info(f"Val: {new_num_samples_val} images")

    # with open(config_file_path, "a") as f:
    #     f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
    #     f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    # limit the datasets to those new numbers
    datasets_as_df["train"] = datasets_as_df["train"][:new_num_samples_train]
    datasets_as_df["valid"] = datasets_as_df["valid"][:new_num_samples_val]

    return datasets_as_df

def get_datasets(path_full_dataset):
    PERCENTAGE_OF_TRAIN = 1.0
    PERCENTAGE_OF_VAL= 1.0
    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
    }

    datasets_as_dfs = {}
    datasets_as_dfs["train"] = pd.read_csv(os.path.join(path_full_dataset, "train.csv"), usecols=usecols, converters=converters)

    # val dataset has additional "reference_report" column
    usecols.append("reference_report")
    datasets_as_dfs["valid"] = pd.read_csv(os.path.join(path_full_dataset, "valid.csv"), usecols=usecols, converters=converters)

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL * total_num_samples_val)

    # log.info(f"Train: {new_num_samples_train} images")
    # log.info(f"Val: {new_num_samples_val} images")

    # with open(config_file_path, "a") as f:
    #     f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
    #     f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_train_dataset, raw_val_dataset

def get_testset(path_full_dataset):
    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
        "reference_report"
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
    }

    datasets_as_dfs = {}
    datasets_as_dfs["test"] = pd.read_csv(os.path.join(path_full_dataset, "test.csv"), usecols=usecols, converters=converters)
    datasets_as_dfs["test-2"] = pd.read_csv(os.path.join(path_full_dataset, "test-2.csv"), usecols=usecols, converters=converters)

    raw_test_dataset = Dataset.from_pandas(datasets_as_dfs["test"])
    raw_test_2_dataset = Dataset.from_pandas(datasets_as_dfs["test-2"])

    return raw_test_dataset, raw_test_2_dataset


class CustomCollator:
    def __init__(self, tokenizer, is_val):
        self.tokenizer = tokenizer
        self.is_val = is_val

    def __call__(self, batch: list[dict[str]]):
        """
        batch is a list of dicts where each dict corresponds to a single image and has the keys:
          - image
          - bbox_coordinates
          - bbox_labels
          - input_ids
          - attention_mask
          - bbox_phrase_exists
          - bbox_is_abnormal
          - bbox_phrases

        For the val and test datasets, we have the additional key:
          - reference_report
        """
        # discard samples from batch where __getitem__ from custom_dataset failed (i.e. returned None)
        # otherwise, whole training loop would stop
        batch = list(filter(lambda x: x is not None, batch))  # filter out samples that are None

        # allocate an empty tensor images_batch that will store all images of the batch
        image_size = batch[0]["image"].size()
        images_batch = torch.empty(size=(len(batch), *image_size))

        # create an empty list image_targets that will store dicts containing the bbox_coordinates and bbox_labels
        image_targets = []
        # for a validation and test batch, create a List[List[str]] that hold the reference phrases (i.e. bbox_phrases) to compute e.g. BLEU scores
        # the inner list will hold all reference phrases for a single image
        bbox_phrases_batch = []
        # allocate an empty tensor region_has_sentence that will store all bbox_phrase_exists tensors of the batch
        bbox_phrase_exists_size = batch[0]["bbox_phrase_exists"].size()  # should be torch.Size([29])
        region_has_sentence = torch.empty(size=(len(batch), *bbox_phrase_exists_size), dtype=torch.bool)

        # allocate an empty tensor region_is_abnormal that will store all bbox_is_abnormal tensors of the batch
        bbox_is_abnormal_size = batch[0]["bbox_is_abnormal"].size()  # should be torch.Size([29])
        region_is_abnormal = torch.empty(size=(len(batch), *bbox_is_abnormal_size), dtype=torch.bool)

        if self.is_val:
            # create a List[str] to hold the reference reports for the images in the batch
            reference_reports = []

        for i, sample_dict in enumerate(batch):
            # remove image tensors from batch and store them in dedicated images_batch tensor
            images_batch[i] = sample_dict.pop("image")

            # remove bbox_coordinates and bbox_labels and store them in list image_targets
            boxes = sample_dict.pop("bbox_coordinates")
            labels = sample_dict.pop("bbox_labels")
            image_targets.append({"boxes": boxes, "labels": labels})

            # remove bbox_phrase_exists tensors from batch and store them in dedicated region_has_sentence tensor
            region_has_sentence[i] = sample_dict.pop("bbox_phrase_exists")

            # remove bbox_is_abnormal tensors from batch and store them in dedicated region_is_abnormal tensor
            region_is_abnormal[i] = sample_dict.pop("bbox_is_abnormal")

            # remove list bbox_phrases from batch and store it in the list bbox_phrases_batch
            bbox_phrases_batch.append(sample_dict.pop("bbox_phrases"))

            if self.is_val:
                # remove reference reports from batch and store it in the list bbox_phrases_batch
                reference_reports.append(sample_dict.pop("reference_report"))

        # if self.pretrain_without_lm_model:
        #     batch = {}
        # else:
        # batch is now a list that only contains dicts with keys input_ids and attention_mask (both of which are List[List[int]])
        # i.e. batch is of type List[Dict[str, List[List[int]]]]
        # each dict specifies the input_ids and attention_mask of a single image, thus the outer lists always has 29 elements (with each element being a list)
        # for sentences describing 29 regions
        # we want to pad all input_ids and attention_mask to the max sequence length in the batch
        # we can use the pad method of the tokenizer for this, however it requires the input to be of type Dict[str, List[List[int]]
        # thus we first transform the batch into a dict with keys "input_ids" and "attention_mask", both of which are List[List[int]]
        # that hold the input_ids and attention_mask of all the regions in the batch (i.e. the outer list will have (batch_size * 29) elements)
        dict_with_ii_and_am = self.transform_to_dict_with_inputs_ids_and_attention_masks(batch)

        # we can now apply the pad method, which will pad the input_ids and attention_mask to the longest sequence in the batch
        # the keys "input_ids" and "attention_mask" in dict_with_ii_and_am will each map to a tensor of shape [(batch_size * 29), (longest) seq_len (in batch)]
        dict_with_ii_and_am = self.tokenizer.pad(dict_with_ii_and_am, padding="longest", return_tensors="pt")

        # treat dict_with_ii_and_am as the batch variable now (since it is a dict, and we can use it to store all the other keys as well)
        batch = dict_with_ii_and_am

        # add the remaining keys and values to the batch dict
        batch["images"] = images_batch
        batch["image_targets"] = image_targets
        batch["region_has_sentence"] = region_has_sentence
        batch["region_is_abnormal"] = region_is_abnormal
        batch["reference_sentences"] = bbox_phrases_batch

        if self.is_val:
            batch["reference_reports"] = reference_reports

        return batch

    def transform_to_dict_with_inputs_ids_and_attention_masks(self, batch):
        dict_with_ii_and_am = {"input_ids": [], "attention_mask": []}
        for single_dict in batch:
            for key, outer_list in single_dict.items():
                for inner_list in outer_list:
                    dict_with_ii_and_am[key].append(inner_list)

        return dict_with_ii_and_am

