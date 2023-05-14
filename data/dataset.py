import os

import torch

import pickle
from tqdm import tqdm


class mbti_classification_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data, data_type="dataset"):
        self._data_list = []

        for pkl_path in tqdm(data, desc=f"Loading files of {data_type}..."):
            with open(os.path.join('dataset/processed', pkl_path), 'rb') as f:
                data_point = pickle.load(f)

                if "input_ids" not in data_point or "attention_mask" not in data_point or "MBTI" not in data_point:
                    continue

                self._data_list.append((data_point["input_ids"],
                                        data_point["attention_mask"],
                                        data_point["MBTI"][0]))

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        return self._data_list[index]
