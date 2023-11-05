from collections import defaultdict

import torch
# czy w papierze chodziło o stale martwe neurony?
class DeadActivationCallback:
    '''
    Gather dead activations
    '''
    def __init__(self):
        self.dead_acts = defaultdict(int)
        self.denoms = defaultdict(int)
        self.idx = 0
        self.is_able = True

    def __call__(self, module, input, output):
        if isinstance(module, torch.nn.ReLU) and self.is_able:
            self.dead_acts[f'per_scalar_{module._get_name()}_{self.idx}'] += torch.sum(output == 0).item()
            self.denoms[f'per_scalar_{module._get_name()}_{self.idx}'] += output.numel()
            cond = torch.all(output == 0, dim=0) if len(output.shape) <= 2 else torch.all(output == 0, dim=0).all(dim=1).all(dim=1)
            self.dead_acts[f'per_neuron_{module._get_name()}_{self.idx}'] += torch.sum(cond).item()
            self.denoms[f'per_neuron_{module._get_name()}_{self.idx}'] += output.size(1)
            self.idx += 1       
        
    def prepare(self):
        self.dead_acts['per_scalar_total'] = sum(self.dead_acts[tag] for tag in self.dead_acts if 'per_scalar' in tag) # rozróżnienie na dwa totale
        self.denoms['per_scalar_total'] = sum(self.denoms[tag] for tag in self.denoms if 'per_scalar' in tag)
        self.dead_acts['per_neuron_total'] = sum(self.dead_acts[tag] for tag in self.dead_acts if 'per_neuron' in tag)
        self.denoms['per_neuron_total'] = sum(self.denoms[tag] for tag in self.denoms if 'per_neuron' in tag)
        new_dead_acts = defaultdict(int)
        for tag in self.dead_acts:
            new_dead_acts[f'dead_activations/{tag}'] = self.dead_acts[tag] / self.denoms[tag]
        self.dead_acts = new_dead_acts
               
    def reset(self):
        self.dead_acts = defaultdict(int)
        self.denoms = defaultdict(int)
        self.idx = 0
        
    def disable(self):
        self.is_able = False
        
    def enable(self):
        self.is_able = True
        
    def get_assets(self):
        return self.dead_acts
        
        
class GatherRepresentationsCallback:
    '''
    Gather layers' representations.
    '''
    def __init__(self, cutoff):
        self.representations = {}
        self.idx = 0
        self.is_able = False
        self.subsampling = defaultdict(lambda: None)
        self.cutoff = cutoff
        self.device = None

    def __call__(self, module, input, output):
        if self.is_able:
            if self.device is None: self.device = output.device
            name = module._get_name() + f'_{"" if self.idx > 9 else "0"}{self.idx}'
            output = output.flatten(start_dim=1)
            
            if name in self.subsampling:
                output = self.adjust_representation(output, name)
            elif output.size(1) > self.cutoff:
                self.subsampling[name] = torch.randperm(output.size(1))[:self.cutoff].sort()[0]
                output = self.adjust_representation(output, name)
            
            self.representations[name] = output
            self.idx += 1
            
    def adjust_representation(self, representation, name):
        representation = torch.index_select(representation, 1, self.subsampling[name].to(self.device))
        return representation
    
    def get_assets(self):
        return self.representations
        
    def reset(self):
        self.representations = {}
        self.idx = 0
        
    def disable(self):
        self.is_able = False
        
    def enable(self):
        self.is_able = True


CALLBACK_TYPE = {
    'dead_relu': DeadActivationCallback,
    'gather_reprs': GatherRepresentationsCallback,
    }
