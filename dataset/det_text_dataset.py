import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class DetTextSet(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset, transforms):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms
        # self.log = log
        self.cxr_wh=json.load(open('../data/mimic-cxr-reports/cxr_wh.json'))

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # get the image_path for potential logging in except block
        image_path = self.tokenized_dataset[index]["mimic_image_file_path"]
        
        

        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            bbox_coordinates = self.tokenized_dataset[index]["bbox_coordinates"]  # List[List[int]]
            bbox_labels = self.tokenized_dataset[index]["bbox_labels"]  # List[int]
            input_ids = self.tokenized_dataset[index]["input_ids"]  # List[List[int]]
            attention_mask = self.tokenized_dataset[index]["attention_mask"]  # List[List[int]]
            bbox_phrases = self.tokenized_dataset[index]["bbox_phrases"]  # List[str]
            bbox_phrase_exists = self.tokenized_dataset[index]["bbox_phrase_exists"]  # List[bool]
            bbox_is_abnormal = self.tokenized_dataset[index]["bbox_is_abnormal"]  # List[bool]

            if self.dataset_name != "train":
                # we only need the reference report during evaluation when computing scores for metrics
                reference_report = self.tokenized_dataset[index]["reference_report"]  # str

            if image_path is None or not os.path.isfile(image_path):
                # only used for local machine debugging (The MIMIC-CXR dataset has a size exceeding 500 GB)
                # print("[WARNING] Pseudo CXR now")
                width, height=self.cxr_wh[image_path]['width'],self.cxr_wh[image_path]['height']
                image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
            else:
                # cv2.imread by default loads an image with 3 channels
                # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


            # apply transformations to image, bbox_coordinates and bbox_labels
            transformed = self.transforms(image=image, bboxes=bbox_coordinates, class_labels=bbox_labels)

            transformed_image = transformed["image"]
            transformed_bbox_coordinates = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]

            sample = {
                "image": transformed_image,
                # "raw_image":image,
                "bbox_coordinates": torch.tensor(transformed_bbox_coordinates, dtype=torch.float),
                # "raw_bbox_coordinates": torch.tensor(bbox_coordinates, dtype=torch.float),
                "bbox_labels": torch.tensor(transformed_bbox_labels, dtype=torch.int64),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "bbox_phrases":bbox_phrases,
                "bbox_phrase_exists": torch.tensor(bbox_phrase_exists, dtype=torch.bool),
                "bbox_is_abnormal": torch.tensor(bbox_is_abnormal, dtype=torch.bool),
            }

            if self.dataset_name != "train":
                sample["reference_report"] = reference_report

        except Exception as e:
            print(f"__getitem__ failed for: {image_path}")
            print(f"Reason: {e}")
            # self.log.error(f"__getitem__ failed for: {image_path}")
            # self.log.error(f"Reason: {e}")
            return None

        return sample