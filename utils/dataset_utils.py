import os
from unicodedata import normalize

import datasets
import pandas as pd
from datasets import Dataset

from literal import DatasetColumns, Folder, RawDataColumns


def to_subchar(string: str) -> str:
    """
    convert unicode subchar
    """
    return normalize("NFKD", string)


def preprocess_img_path(path: str):
    if not path.startswith(Folder.data):
        path = path.replace("./", Folder.data)
    return path


def get_dataset(csv_path: os.PathLike, is_sub_char=False) -> Dataset:
    """
    csv img infomation to dataset
    is_sub_char: "snunlp/KR-BERT-char16424"와 같은 sub_char tokenizer일 경우 True, 일반적인 토크나이저일 경우에는 False
    feature: pixel_values(PIL image), labels(str)
    """
    df = pd.read_csv(csv_path)

    data_dict = {DatasetColumns.pixel_values: df[RawDataColumns.img_path].tolist()}

    if RawDataColumns.label in df.columns:
        if is_sub_char:
            df[RawDataColumns.label] = df[RawDataColumns.label].apply(to_subchar)
        data_dict[DatasetColumns.labels] = df[RawDataColumns.label].tolist()

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(DatasetColumns.pixel_values, datasets.Image())
    return dataset
