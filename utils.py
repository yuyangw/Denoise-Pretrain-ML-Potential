import math


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config['warmup_epochs']:
        lr = (config['lr'] - config['min_lr']) * epoch / config['warmup_epochs'] + config['min_lr']
    elif epoch < config['warmup_epochs'] + config['patience_epochs']:
        lr = config['lr']
    else:
        prev_epochs = config['warmup_epochs'] + config['patience_epochs']
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - prev_epochs) / (config['epochs'] - prev_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
