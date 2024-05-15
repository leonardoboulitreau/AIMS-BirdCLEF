import torch

def get_optimizer(model, training_args):
    if training_args['optimizer'] == 'adam':
        opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=training_args['lr'],
                weight_decay=training_args['weight_decay']
            )
        return opt
    else:
        raise NotImplementedError