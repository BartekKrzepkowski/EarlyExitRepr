from typing import List, Dict, Any

import torch

from src.utils.utils_model import infer_flatten_dim
from src.utils import common


class Dummy(torch.nn.Module):
    def __init__(self, layers_dim: List[int]):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim1, hidden_dim2)
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-1], layers_dim[1:])
        ])
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

    def forward_generator(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
            x = yield x, None
        y = self.final_layer(x)
        _ = yield x, y

class MLPwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                torch.nn.BatchNorm1d(hidden_dim2),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x
    

class MLPwithDropout(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(hidden_dim1, hidden_dim2),
                                torch.nn.Dropout(p=0.15),
                                common.ACT_NAME_MAP[activation_name]())
            for hidden_dim1, hidden_dim2 in zip(layers_dim[:-2], layers_dim[1:-1])
        ])
        self.final_layer = torch.nn.Linear(layers_dim[-2], layers_dim[-1])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
    
class SimpleCNNwithNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
class SimpleCNNwithGroupNorm(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.GroupNorm(layer_dim2, layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.GroupNorm(layer_dim2, layer_dim2),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x


class SimpleCNNwithDropout(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.Dropout(0.05),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    
    
class SimpleCNNwithNormandDropout(torch.nn.Module):
    def __init__(self, layers_dim: List[int], activation_name: str, conv_params: Dict[str, Any]):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(layer_dim1, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.Conv2d(layer_dim2, layer_dim2, 3, padding=1),
                                torch.nn.BatchNorm2d(layer_dim2),
                                torch.nn.Dropout(0.15),
                                common.ACT_NAME_MAP[activation_name](),
                                torch.nn.MaxPool2d(2, 2))
            for layer_dim1, layer_dim2 in zip(layers_dim[:-3], layers_dim[1:-2])
        ])
        flatten_dim = infer_flatten_dim(conv_params, layers_dim[-3])
        # napisz wnioskowanie spłaszczonego wymiaru
        self.final_layer = torch.nn.Sequential(torch.nn.Linear(flatten_dim, layers_dim[-2]), common.ACT_NAME_MAP[activation_name](),
                                               torch.nn.BatchNorm1d(layers_dim[-2]),
                                               torch.nn.Dropout(0.05),
                                               torch.nn.Linear(layers_dim[-2], layers_dim[-1]))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.final_layer(x)
        return x
    