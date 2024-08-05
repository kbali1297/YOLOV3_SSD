import torch
import copy
from torch.nn.utils import prune


def prune_model(model, amount, dim=0, norm=2):

    for i, (name, modules) in enumerate(model.named_modules()):
        if name=='vgg' or name=='extras':
            for j, (sub_name, sub_module) in enumerate(modules.named_modules()):
                #print(f'{j} sub_name:{sub_name} sub_module:{sub_module}')
                if isinstance(sub_module, torch.nn.Conv2d):
                    prune.ln_structured(sub_module, 'weight', amount, n=norm, dim=dim)

    sum_pruned = 0
    for name, buffer in model.named_buffers():
        sum_pruned += (buffer==0).sum()

    total_params = 0
    for p in model.parameters():
        try: total_params += p.numel()
        except: pass
    
    return model, (total_params - sum_pruned)/total_params

def remove_pruning(model):
     
    for i, (name, modules) in enumerate(model.named_modules()):
        if name=='vgg' or name=='extras':
            for j, (sub_name, sub_module) in enumerate(modules.named_modules()):
                #print(f'{j} sub_name:{sub_name} sub_module:{sub_module}')
                if isinstance(sub_module, torch.nn.Conv2d):
                    prune.remove(sub_module, 'weight')

    return model

    