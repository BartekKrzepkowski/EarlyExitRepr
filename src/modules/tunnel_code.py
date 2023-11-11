import os
import logging
from abc import abstractmethod

import numpy as np
import torch
import matplotlib.pyplot as plt


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
    def __init__(self, model, loader, layers=None, rpath='.', MAX_REPR_SIZE=8000):
        self.model = model
        self.loader = loader
        self.layers_to_analyze = layers if layers is not None else [n for n, m in model.named_modules()]
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

    def _spectra_hook(self, name):
        def spectra_hook(model, input, output):
            representation_size = int(output.numel()/output.shape[0])
            output = output.flatten(1)
            if representation_size > self.MAX_REPR_SIZE:                
                output = output[:, np.random.choice(representation_size, self.MAX_REPR_SIZE, replace=False)]
            self.representations[name] = self.representations.get(name, []) + [output]
        return spectra_hook

    def _insert_hooks(self):
        for name, layer in self.model.named_modules():
            if name in self.layers_to_analyze:
                self.handels.append(layer.register_forward_hook(self._spectra_hook(name)))

    @torch.no_grad()
    def collect_representations(self):
        self.model.eval()
        with torch.no_grad():
            for x, *_ in self.loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                _ = self.model(x)
        for name, rep in self.representations.items():
            self.representations[name] = torch.cat(rep, dim=0).cpu().detach()
        for handle in self.handels:
            handle.remove()
        return self.representations

    def analysis(self, step, scope, phase):
        prefix = 'ranks_representations'
        postfix = f'____{scope}____{phase}'
        if len(self.representations) == 0:
            logging.info(f'Number of matrices: {9991}')
            self.collect_representations()
        logging.info(f'Number of matrices: {9992}')
        evaluators = {}
        for name, rep in self.representations.items():
            name_dict = f'{prefix}/{name}{postfix}'
            rep = torch.cov(rep.T)
            evaluators[name_dict] = torch.linalg.matrix_rank(rep)
        evaluators['steps/tunnel'] = step
        self.logger.log_scalars(evaluators, step)
        self.clean_up()

    # def plot(self, name):
    #     effective_rank = self.result["rank"]
    #     fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    #     axs.plot(list(effective_rank.keys()), list(effective_rank.values()), "o-")
    #     plt.savefig(os.path.join(self.rpath, name + ".png"), dpi=500)
    #     plt.close()