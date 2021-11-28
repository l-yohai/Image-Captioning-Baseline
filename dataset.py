import torch
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import json


class ImageCaptioningDataset(Dataset):
    def __init__(self, root_dir, df, feature_extractor, tokenizer, max_target_length=512):
        self.root_dir = root_dir
        self.df = df
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_target_length

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_path = self.df["file_path"][idx]
        caption = self.df["captions"][idx]

        image = Image.open(self.root_dir + "/" + image_path).convert("RGB")
        pixel_values = self.feature_extractor(
            image, return_tensors="pt").pixel_values

        tokenized_caption = self.tokenizer(
            caption, padding="max_length", max_length=self.max_length).input_ids
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(tokenized_caption),
        }
        return encoding


def load_dataset(root_dir, dataset_path, feature_extractor, tokenizer, max_target_length=512):
    with open(dataset_path, "r") as f:
        coco_dataset = json.load(f)

    captions = []
    file_paths = []

    for i in range(len(coco_dataset)):
        captions.append(coco_dataset[i]["captions"][0])
        file_paths.append(coco_dataset[i]["file_path"])

    df = pd.DataFrame(data={"captions": captions, "file_path": file_paths})

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()

    train_dataset = ImageCaptioningDataset(
        root_dir, train_df, feature_extractor, tokenizer, max_target_length
    )
    val_dataset = ImageCaptioningDataset(
        root_dir, val_df, feature_extractor, tokenizer, max_target_length)

    return train_dataset, val_dataset
