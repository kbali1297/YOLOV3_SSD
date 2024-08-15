import torch, argparse
from utils.models import SSD
from utils.prune import *
import shutil

parser = argparse.ArgumentParser(description='CmdLine Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(  '--load_pruned_model', default='SSD_pruned_0.22.pth', type=str)

pargs = parser.parse_args()
params = {}
params.update(vars(pargs))

#model_SSD = SSD("./config/ssd-kitticopy_.cfg", 1).to("cuda:0")
model_SSD = SSD("./config/ssd-kitti.cfg", 1).to("cuda:0")
model_SSD, _ = prune_model(model_SSD, 0, 0, 2)
model_SSD.load_state_dict(torch.load(params['load_pruned_model'])) # 52% pruned SSD
print('model successfully loaded')

non_pruned_filter_idxs = {}
for i, (name, modules) in enumerate(model_SSD.named_modules()):
    if name=='vgg' or name=='extras':
        for j, (sub_name, sub_module) in enumerate(modules.named_buffers()):
            #print(f'{j} sub_name:{sub_name} sub_module:{sub_module}')
            non_pruned_filter_idxs[f'{name}_{sub_name}'] = []
            for dim_ in range(sub_module.shape[0]):
                if torch.sum(sub_module[dim_, ...]) != 0:
                    non_pruned_filter_idxs[f'{name}_{sub_name}'].append(dim_)

base_dict = {'300':[], '512':[]}
extras_dict = {'300':[], '512':[]}
for key, val in non_pruned_filter_idxs.items():
    print(f'{key}: {len(val)}')
    layer_name, layer_num = key[:-12].split('_')
    if layer_name == 'vgg':
        base_dict['300'].append(len(val))
    elif layer_name == 'extras':
        extras_dict['300'].append(len(val))

base_pool = ['M', 'M', 'C', 'M']
extras_pool = ['S', 'S']

base_, extras_ = {'300':[], '512':[]}, {'300':[], '512':[]}
b_i, e_i = 0,0
b_j, e_j = 0,0
base_len = len(base_dict['300']) + len(base_pool)
for i in range(base_len):
    if i in [2,5,9,13]:
        base_['300'].append(base_pool[b_j])
        b_j += 1
    else:
        base_['300'].append(base_dict['300'][b_i])
        b_i += 1

extras_len = len(extras_dict['300']) + len(extras_pool)
for i in range(extras_len):
    if i in [1,4]:
        extras_['300'].append(extras_pool[e_j])
        e_j += 1
    else:
        extras_['300'].append(extras_dict['300'][e_i])
        e_i += 1

print(model_SSD.eval())

## Prepare config file
amount_unpruned = params['load_pruned_model'].split('_')[-1][:4]
new_config_fname = f'config/ssd-kitti_{amount_unpruned}.cfg'

with open('config/ssd-kitti.cfg', 'r') as file:
    data = file.readlines()

data[22] = str(data[22][:5]) + str(base_) + "\n"
data[23] = str(data[23][:7]) + str(extras_) + "\n"

with open(new_config_fname, 'w') as file:
    file.writelines(data)

# Pruned Model : Has only non zero filters and weights
model_SSD_p = SSD(new_config_fname, 1).to("cuda:0")

print(model_SSD.state_dict().keys())

## Copy non zero weights in the vgg, extras, loc and conf conv layers 
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
                        
    
# loc and conf layers

input_layers = ['vgg_21', 'vgg_33', 'extras_1', 'extras_3', 'extras_5', 'extras_7']

for i, inp_layer_name in enumerate(input_layers):
    with torch.no_grad():
        model_SSD_p.state_dict()[f'loc.{i}.weight'].copy_(model_SSD.state_dict()[f'loc.{i}.weight']\
                                                                        [:, non_pruned_filter_idxs[f'{inp_layer_name}.weight_mask'], ...])
        model_SSD_p.state_dict()[f'conf.{i}.weight'].copy_(model_SSD.state_dict()[f'conf.{i}.weight']\
                                                                        [:, non_pruned_filter_idxs[f'{inp_layer_name}.weight_mask'], ...])

pruned_model_wt = model_SSD.state_dict()['vgg.19.weight_orig'] * model_SSD.state_dict()['vgg.19.weight_mask']
new_model_wt = model_SSD_p.state_dict()['vgg.19.weight']

torch.save(model_SSD_p.state_dict(), f'ssd_compressed_{amount_unpruned}.pth')
## Check whether non zero weights match
# print(f'pruned_model_wt: {pruned_model_wt}')
# print(f'new_model_wt: {new_model_wt}')

# print(f'pruned_model_wt shape: {pruned_model_wt.shape}')
# print(f'new_model_wt shape: {new_model_wt.shape}')






