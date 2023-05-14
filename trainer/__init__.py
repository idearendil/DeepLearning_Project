from .trainer import mbti_classification


def get_trainer(args, iteration, x, static, y, model, device, scheduler, optimizer, criterion, flow_type=None):
    if args.trainer == "mbti_classification":
        model, iter_loss = mbti_classification(args, iteration, x, y, model, device, scheduler, optimizer, criterion, flow_type)
    else:
        print("Selected Trainer is Not Prepared Yet...")
        exit(1)

    return model, iter_loss
