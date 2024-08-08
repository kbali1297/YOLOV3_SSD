import torch
from utils.models import SSD
from utils.prune import *

#model_SSD = SSD("./config/ssd-kitticopy_.cfg", 1).to("cuda:0")
model_SSD = SSD("./config/ssd-kitti.cfg", 1).to("cuda:0")
model_SSD, _ = prune_model(model_SSD, 0, 0, 2)
model_SSD.load_state_dict(torch.load("SSD_pruned_0.41_.pth"))
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

#print(non_pruned_filter_idxs)
for key, val in non_pruned_filter_idxs.items():
    print(f'{key}: {len(val)}')

print(model_SSD.eval())

## Pruned Model (Has more filters)
model_SSD_p = SSD("./config/ssd-kitti_copy.cfg", 1).to("cuda:0")
#model_SSD_p.load_state_dict(torch.load("SSD_pruned.pth"))

print(model_SSD.state_dict().keys())
#non_pruned_conv = model_SSD.state_dict()['vgg.0.weight_orig'][non_pruned_filter_idxs['vgg_0.weight_mask'], ...]
#non_pruned_conv_mask = model_SSD.state_dict()['vgg.0.weight_orig'][non_pruned_filter_idxs['vgg_0.weight_mask'], ...]
#non_pruned_conv_bias = model_SSD.state_dict()['vgg.0.bias'][non_pruned_filter_idxs['vgg_0.weight_mask'], ...]
#print(non_pruned_conv)
#print(non_pruned_conv_bias)
#print(non_pruned_conv.shape)
#print(non_pruned_conv_bias.shape)

prev_layer_name = None
for dict_idx in range(len(non_pruned_filter_idxs.items())):
    layer_name = list(non_pruned_filter_idxs.keys())[dict_idx]
    layer_type, layer_idx = layer_name.split('.')[0].split('_')
    
    model_layer_key = f'{layer_type}.{layer_idx}'
    idx_dict_key = f'{layer_type}_{layer_idx}.weight_mask'
        

    with torch.no_grad():
        if model_layer_key == 'vgg.0':
            model_SSD_p.state_dict()[f'{model_layer_key}.weight'].copy_(model_SSD.state_dict()[f'{model_layer_key}.weight_orig']\
                                            [non_pruned_filter_idxs[idx_dict_key], ...])
                        
        else:
            prev_layer_name = list(non_pruned_filter_idxs.keys())[dict_idx-1]
            prev_layer_type, prev_layer_idx = prev_layer_name.split('.')[0].split('_')
        
            model_prev_layer_key = f'{prev_layer_type}.{prev_layer_idx}'
            prev_idx_dict_key = f'{prev_layer_type}_{prev_layer_idx}.weight_mask'
            
            model_SSD_p.state_dict()[f'{model_layer_key}.weight'].copy_(model_SSD.state_dict()[f'{model_layer_key}.weight_orig']\
                                            [non_pruned_filter_idxs[idx_dict_key], ...]\
                                                [:, non_pruned_filter_idxs[prev_idx_dict_key], ...])
                        
            
            # tensor_to_be_copied = model_SSD.state_dict()[f'{model_layer_key}.weight_orig']\
            #                                 [non_pruned_filter_idxs[idx_dict_key], ...]\
            #                                 [:, non_pruned_filter_idxs[prev_idx_dict_key], ...]
            # tensor_set_ascopy_ = model_SSD_p.state_dict()[f'{model_layer_key}.weight']
            # if model_layer_key == "vgg.19":
            #     print("hey!")
    
# loc and conf layers

input_layers = ['vgg_21', 'vgg_33', 'extras_1', 'extras_3', 'extras_5', 'extras_7']

for i, inp_layer_name in enumerate(input_layers):
    with torch.no_grad():
        model_SSD_p.state_dict()[f'loc.{i}.weight'].copy_(model_SSD.state_dict()[f'loc.{i}.weight']\
                                                                        [:, non_pruned_filter_idxs[f'{inp_layer_name}.weight_mask'], ...])
        model_SSD_p.state_dict()[f'conf.{i}.weight'].copy_(model_SSD.state_dict()[f'conf.{i}.weight']\
                                                                        [:, non_pruned_filter_idxs[f'{inp_layer_name}.weight_mask'], ...])
    
# model_SSD_p.state_dict()['L2Norm.weight'] = model_SSD.state_dict()['L2Norm.weight']\
#                                                 [non_pruned_filter_idxs['vgg_21.weight_mask'],...]

output_p, *_ = model_SSD_p(random_input)

torch.save(model_SSD_p.state_dict(), "ssd_compressed.pth")
print('Pruned Model op:')
print(output)

print('Model with Pruned model pasted weights op:')
print(output_p)

pruned_model_wt = model_SSD.state_dict()['vgg.19.weight_orig'] * model_SSD.state_dict()['vgg.19.weight_mask']
new_model_wt = model_SSD_p.state_dict()['vgg.19.weight']
#model_SSD_p.state_dict()['vgg.19.weight'][0,...] = 0 
print(f'pruned_model_wt: {pruned_model_wt}')
print(f'new_model_wt: {new_model_wt}')

print(f'pruned_model_wt shape: {pruned_model_wt.shape}')
print(f'new_model_wt shape: {new_model_wt.shape}')






