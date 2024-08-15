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

    sum_not_pruned = 0
    filter_channel_list = torch.arange(3) # Consider all channels coming from the input
    for i, (name, buffer) in enumerate(model.named_buffers()):
        # Since the output of the current buffer would also have zero channels,
        # the following conv filters do not need to have non zero input channels at 
        # those dimension locations
        sum_not_pruned += (buffer[:, filter_channel_list, ...]==1).sum()
        # computing which filters were not zeroes out to consider only
        # those input channels in the coming conv filter
        filter_channel_list = []
        for channel in range(buffer.shape[0]):
            if buffer[channel,...].sum() !=0:
                filter_channel_list.append(channel)
        


    total_params = 0
    for p in model.parameters():
        try: total_params += p.numel()
        except: pass
    
    return model, sum_not_pruned/total_params

def remove_pruning(model):
     
    for i, (name, modules) in enumerate(model.named_modules()):
        if name=='vgg' or name=='extras':
            for j, (sub_name, sub_module) in enumerate(modules.named_modules()):
                #print(f'{j} sub_name:{sub_name} sub_module:{sub_module}')
                if isinstance(sub_module, torch.nn.Conv2d):
                    prune.remove(sub_module, 'weight')

    return model

    