"""
data_preprocess file
"""


import random

import os

from data.collate_fn import collate_txt
from data.dataset import mbti_classification_Dataset
from torch.utils.data import DataLoader


def get_data_loader(args):

    print("Initializing Data Loader and Datasets")

    last_digits = [i for i in range(10)]
    random.shuffle(last_digits)

    train_ids = last_digits[:6]
    val_ids = last_digits[6:8]
    test_ids = last_digits[8:]

    train_data_list = []
    val_data_list = []
    test_data_list = []

    file_dir = os.listdir(os.path.join(os.getcwd(), 'dataset/processed'))
    for data_file in file_dir:
        data_session_id = int(data_file.split("/")[-1][4:6])
        if data_session_id in train_ids:
            train_data_list.append(data_file)
        elif data_session_id in val_ids:
            val_data_list.append(data_file)
        elif data_session_id in test_ids:
            test_data_list.append(data_file)

    if args.trainer == "mbti_classification":
        train_data = mbti_classification_Dataset(args, data=train_data_list, data_type="training dataset")
        val_data = mbti_classification_Dataset(args, data=val_data_list, data_type="validation dataset")
        test_data = mbti_classification_Dataset(args, data=test_data_list, data_type="testing dataset")

    print(f"Total of {train_data.__len__()} data points intialized in Training Dataset...")
    print(f"Total of {val_data.__len__()} data points intialized in Validation Dataset...")
    print(f"Total of {test_data.__len__()} data points intialized in Testing Dataset...")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_txt)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_txt)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True, collate_fn=collate_txt)

    return train_loader, val_loader, test_loader
