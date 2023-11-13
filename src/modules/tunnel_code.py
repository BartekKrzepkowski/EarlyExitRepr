import os
import logging
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch


class BaseAnalysis:
    def export(self, name):
        torch.save(self.result, os.path.join(self.rpath, name + ".pt"))

    def clean_up(self):
        for attr in self.attributes_on_gpu:
            try:
                a = getattr(self, attr)
                a.to("cpu")
                del a
            except AttributeError:
                pass
        del self
        torch.cuda.empty_cache()

    @abstractmethod
    def analysis(self):
        pass

    @abstractmethod
    def plot(self, path):
        pass


class RepresentationsSpectra(BaseAnalysis):
    def __init__(self, model, loader, modules_list, layers=None, rpath='.', MAX_REPR_SIZE=8000):
        self.model = model
        self.loader = loader
        self.layers_to_analyze = layers if layers is not None else [n for n, m in model.named_modules()]
        self.modules_list = modules_list
        self.handels = []
        self._insert_hooks()
        self.representations = {}
        self.rpath = rpath
        self.MAX_REPR_SIZE = MAX_REPR_SIZE
        # os.makedirs(self.rpath, exist_ok=True)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()
        self.attributes_on_gpu = ["model"]
        self.logger = None
        self.is_able = False

    def _spectra_hook(self, name):
        def spectra_hook(model, input, output):
            if self.is_able:
                representation_size = int(output.numel()/output.shape[0])
                output = output.flatten(1)
                if representation_size > self.MAX_REPR_SIZE:                
                    output = output[:, np.random.choice(representation_size, self.MAX_REPR_SIZE, replace=False)]
                self.representations[name] = self.representations.get(name, []) + [output]
        return spectra_hook

    def _insert_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.layers_to_analyze:
                if any(isinstance(module, module_type) for module_type in self.modules_list):
                    self.handels.append(module.register_forward_hook(self._spectra_hook(name)))
                
    def disable(self):
        self.is_able = False
        
    def enable(self):
        self.is_able = True

    @torch.no_grad()
    def collect_representations(self):
        self.model.eval()
        with torch.no_grad():
            for x, *_ in self.loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                _ = self.model(x)
        for name, rep in self.representations.items():
            self.representations[name] = torch.cat(rep, dim=0).detach()
        # for handle in self.handels:
        #     handle.remove()
        self.model.train()
        return self.representations

    def analysis(self, step, scope, phase):
        prefix = 'ranks_representations'
        postfix = f'____{scope}____{phase}'
        if len(self.representations) == 0:
            # logging.info(f'Number of matrices: {9991}')
            self.collect_representations()
        # logging.info(f'self.MAX_REPR_SIZE: {self.MAX_REPR_SIZE}')
        evaluators = {}
        for name, rep in self.representations.items():
            name_dict = f'{prefix}/{name}{postfix}'
            print(name_dict)
            rep = torch.cov(rep.T)
            # rep = rep.T @ rep
            evaluators[name_dict] = torch.linalg.matrix_rank(rep).item()
            
        self.plot(evaluators, prefix, postfix)
        evaluators['steps/tunnel'] = step
        self.logger.log_scalars(evaluators, step)
        self.representations = {}
        torch.cuda.empty_cache()
        # self.clean_up()

    def plot(self, evaluators, prefix, postfix):
        plot_name = f'{prefix}_plots/{postfix}'
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.plot(list(range(len(evaluators))), list(evaluators.values()), "o-")
        # print(list(evaluators.keys()))
        # Dodawanie tytułu i etykiet osi
        axs.set_title("Rank Across Layers")  # Dodaj tytuł wykresu
        axs.set_xlabel("Layer")  # Dodaj etykietę dla osi X
        axs.set_ylabel("Represenation Rank")  # Dodaj etykietę dla osi Y
        plot_images = {plot_name: fig}
        self.logger.log_plots(plot_images)
        # plt.savefig(os.path.join(self.rpath, name + ".png"), dpi=500)
        plt.close()