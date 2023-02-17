from dataclasses import dataclass


@dataclass
class Folder:
    data = "./data/"
    data_preprocess = "./data/preprocess/"


@dataclass
class RawDataColumns:
    img_path = "img_path"
    label = "label"
    length = "length"
    seq_probs = "seq_probs"


@dataclass
class DatasetColumns:
    pixel_values = "pixel_values"
    labels = "labels"
