import json
import os

import pandas as pd

from literal import Folder, RawDataColumns
from utils.dataset_utils import to_subchar

train_df = pd.read_csv(os.path.join(Folder.data, "train.csv"))
labels = train_df[RawDataColumns.label].to_list()
labels = list(map(lambda x: to_subchar(x), labels))

char_vocab = set()
for label in labels:
    for charator in label:
        char_vocab.add(charator)

vocab_list = list(char_vocab)
vocab_list.sort()
vocab_dict = {}
vocab_dict["<pad>"] = 0
vocab_dict["<unk>"] = 1
vocab_dict["<s>"] = 2
vocab_dict["</s>"] = 3
for charator in vocab_list:
    vocab_dict[charator] = len(vocab_dict)

with open("config/vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file, indent=4, ensure_ascii=False)
