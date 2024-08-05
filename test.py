import torch
from utils.models import SSD
from utils.prune import *

#model_SSD = SSD("./config/ssd-kitti_copy.cfg", 1).to("cuda:0")
model_SSD = SSD("./config/ssd-kitti.cfg", 1).to("cuda:0")
model_SSD, _ = prune_model(model_SSD, 0, 0, 2)
model_SSD.load_state_dict(torch.load("ssd.pth"))
print('model successfully loaded')
random_input = torch.rand(1,3,300,300).to("cuda:0")
output, *_ = model_SSD(random_input)
print('model successful in processing forward')
#model_SSD.load_state_dict(torch.load("ssd.pth"))

non_pruned_filter_idxs = {}
for i, (name, modules) in enumerate(model_SSD.named_modules()):
    if name=='vgg' or name=='extras':
        for j, (sub_name, sub_module) in enumerate(modules.named_buffers()):
            #print(f'{j} sub_name:{sub_name} sub_module:{sub_module}')
            non_pruned_filter_idxs[f'{name}_{sub_name}'] = []
            for dim_ in range(sub_module.shape[0]):
                if torch.sum(sub_module[dim_, ...]) != 0:
                    non_pruned_filter_idxs[f'{name}_{sub_name}'].append(dim_)
            #if isinstance(sub_module, torch.nn.Conv2d):
            #    prune.ln_structured(sub_module, 'weight', amount, n=norm, dim=dim)

print(non_pruned_filter_idxs)
for key, val in non_pruned_filter_idxs.items():
    print(f'{key}: {len(val)}')

## Pruned Model (Has more filters)
model_SSD_p = SSD("./config/ssd-kitti_copy.cfg", 1).to("cuda:0")
#model_SSD_p.load_state_dict(torch.load("SSD_pruned.pth"))

print(model_SSD.state_dict().keys())
non_pruned_conv = model_SSD.state_dict()['vgg.0.weight_orig'][non_pruned_filter_idxs['vgg_0.weight_mask'], ...]
#non_pruned_conv_mask = model_SSD.state_dict()['vgg.0.weight_orig'][non_pruned_filter_idxs['vgg_0.weight_mask'], ...]
non_pruned_conv_bias = model_SSD.state_dict()['vgg.0.bias'][non_pruned_filter_idxs['vgg_0.weight_mask'], ...]
print(non_pruned_conv)
print(non_pruned_conv_bias)
print(non_pruned_conv.shape)
print(non_pruned_conv_bias.shape)

for i, (key, non_prune_channels) in enumerate(non_pruned_filter_idxs.items()):
    layer_type, layer_idx = key.split('.')[0].split('_')
    layer_idx = int(layer_idx)
    
    model_layer_key = f'{layer_type}.{layer_idx}'
    idx_dict_key = f'{layer_type}_{layer_idx}.weight_mask'

    model_SSD_p.state_dict()[f'{model_layer_key}.weight'] = \
                    torch.nn.Parameter(model_SSD.state_dict()[f'{model_layer_key}.weight_orig']\
                                       [non_pruned_filter_idxs[idx_dict_key], ...])
   
    model_SSD_p.state_dict()[f'{model_layer_key}.bias'] = \
                    torch.nn.Parameter(model_SSD.state_dict()[f'{model_layer_key}.bias']\
                                       [non_pruned_filter_idxs[idx_dict_key], ...] )
    
for i in range(6):

    model_SSD_p.state_dict()[f'loc.{i}.weight'] = model_SSD.state_dict()[f'loc.{i}.weight']\
                                                    [:,non_pruned_filter_idxs['vgg_21.weight_mask'], ...]
    model_SSD_p.state_dict()[f'conf.{i}.weight'] = model_SSD.state_dict()[f'conf.{i}.weight']\
                                                    [:,non_pruned_filter_idxs['vgg_33.weight_mask'], ...]
    
    model_SSD_p.state_dict()[f'loc.{i}.bias'] = model_SSD.state_dict()[f'loc.{i}.bias']
    model_SSD_p.state_dict()[f'conf.{i}.bias'] = model_SSD.state_dict()[f'conf.{i}.bias']
    
model_SSD_p.state_dict()['L2Norm.weight'] = model_SSD.state_dict()['L2Norm.weight']\
                                                [non_pruned_filter_idxs['vgg_21.weight_mask'],...]

output_p, *_ = model_SSD_p(random_input)

torch.save(model_SSD_p.state_dict(), "ssd_pruned_pasted.pth")
#print('Pruned Model op:')
#print(output)
#
#print('Model with Pruned model pasted weights op:')
#print(output_p)
    





