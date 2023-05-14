import torch
import torch.nn as nn


def mbti_classification(args, iteration, x_txt, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    x_txt = x_txt.type(torch.FloatTensor).to(device)
    y = y.type(torch.LongTensor).to(device)

    if flow_type == "train":
        optimizer.zero_grad()
        output = model(x_txt)
        output = output.squeeze()

        loss = criterion(output, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        scheduler.step()

    elif flow_type == "val":
        output = model(x_txt)
        output = output.squeeze()
        loss = criterion(output, y)
        return output, loss.item()

    else:
        output = model(x_txt)
        output = output.squeeze()
        return output, y

    return model, loss.item()
