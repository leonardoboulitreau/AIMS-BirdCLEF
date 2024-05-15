from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def get_scheduler(optimizer, training_args):
    if training_args['scheduler'] == 'cosine':
        lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=training_args['epochs'],
                    T_mult=1,
                    eta_min=1e-6,
                    last_epoch=-1
                )
        return lr_scheduler
    else:
        raise NotImplementedError
    