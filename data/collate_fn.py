"""
collate_fn file
"""

import torch


def collate_txt(train_data):
    x_batch_id = []
    x_batch_attention = []
    y_batch = []

    for data_point in train_data:
        x_batch_id.append(data_point[0])
        x_batch_attention.append(data_point[1])
        y_batch.append(data_point[2])

    x_ids = torch.stack(x_batch_id)
    x_attention_mask = torch.stack(x_batch_attention)
    y = torch.tensor(y_batch)

    return (x_ids, x_attention_mask), y
