import logging
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.common import ACT_NAME_MAP
from src.modules.heads import StandardHead
from src.modules.metrics import acc_metric

# TODO: 
# Ujednolić nazewnictwo

class SDN(torch.nn.Module):
    # assumption: wiem ile sieć ma możliwych wyjść i w których z nich chce dołączyć klasyfikatory, warstwy wewnętrzne to cnn
    def __init__(self, backbone, criterion, ic_idxs, confidence_threshold, sample, is_model_frozen: bool, prob_conf: bool):
        super().__init__()
        self.device = next(backbone.parameters()).device
        self.n_ics = len(ic_idxs)
        
        self.backbone = backbone
        self.criterion_ic = criterion
        self.ic_idxs = ic_idxs
        self.confidence_threshold = confidence_threshold
        self.is_model_frozen = is_model_frozen
        self.prob_conf = prob_conf
        
        self.attach_heads(sample=sample.to(self.device))
        
        
    def attach_heads(self, sample):
        # assert all(idx < self.nb_of_exits - 1 for idx in self.ic_idxs), "Możemy doczepiać heady do wszystkich prócz ostatniego wyjścia"
        fg = self.backbone.forward_generator(sample)
        input_dims = []
        
        i, x, y = 0, None, None
        while y is None:
            x, y = fg.send(x)
            if i in self.ic_idxs:
                input_dims.append(x.size(1))
            i += 1
        
        num_classes = y.size(-1)
        self.nb_of_exits = i - 2  # "Możemy doczepiać heady do wszystkich prócz ostatniego wyjścia"
            
        self.internal_classifiers = torch.nn.ModuleList([StandardHead(in_channels, num_classes=num_classes, pool_size=4)
            for in_channels in input_dims]).to(self.device)
        
        self.ic_idxs += ['backbone_exit']
        
    # def run_train(self, x_true, y_true):
    #     '''
    #     :param x_true: data input
    #     :param y_true: data label
    #     :return:
    #     '''
    #     internal_representations, y_pred = self.get_internal_representation(x_true)
        
    #     evaluators = defaultdict(float)
    #     logits = []
    #     for i, repr_i in enumerate(internal_representations):
    #         logit_i = self.internal_classifiers[i](repr_i)
    #         logits.append(logit_i)

    #     logits.append(y_pred)

    #     loss_overall = 0.0
    #     for i, logit_i in enumerate(logits):
    #         ce_loss_i = self.criterion_ic(logit_i, y_true)
    #         evaluators[f'internal_classifier_loss/{self.ic_idxs[i]}'] = ce_loss_i.item()
    #         evaluators[f'internal_classifier_acc/{self.ic_idxs[i]}'] = acc_metric(logit_i, y_true)
    #         loss_overall += 0 if ((i + 1) == len(logits) and self.is_model_frozen) else ce_loss_i

    #     evaluators['internal_classifier_loss/overall_loss'] = loss_overall.item()
    #     evaluators['internal_classifier_acc/overall_acc'] = evaluators[f'internal_classifier_acc/{self.ic_idxs[i]}']

    #     return loss_overall, evaluators
    
    
    def forward(self, x_true, y_true, scope):
        # TODO: confusion metric, inconsistency measure
        '''
        :param x_true: data input
        :param y_true: data label
        :return:
        '''
        evaluators = defaultdict(float)
        logits = []
        fg, repr_i, head_idx = self.backbone.forward_generator(x_true), None, 0
        for i in range(self.nb_of_exits):
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            if i not in self.ic_idxs: continue
            logit_i = self.internal_classifiers[head_idx](repr_i)
            logits.append(logit_i)
            head_idx += 1

        _, logit_main = fg.send(fg.send(repr_i)[0])
        logits.append(logit_main)

        loss_overall = 0.0
        for i, logit_i in enumerate(logits):
            losses_i = self.criterion_ic(logit_i, y_true)
            evaluators[f'internal_classifier_loss/{self.ic_idxs[i]}____{scope}'] = losses_i[1]['loss']
            evaluators[f'internal_classifier_acc/{self.ic_idxs[i]}____{scope}'] = losses_i[1]['acc']
            loss_overall += 0 if ((i + 1) == len(logits) and self.is_model_frozen) else losses_i[0]

        evaluators[f'internal_classifier_loss/overall_loss____{scope}'] = loss_overall.item()
        evaluators[f'internal_classifier_acc/overall_acc____{scope}'] = evaluators[f'internal_classifier_acc/{self.ic_idxs[i]}____{scope}']
        # unbounded score - needed to normalize
        p_final = F.softmax(logits[-1], dim=1)
        kl_div_confusion = sum(F.kl_div(F.log_softmax(logit_i, dim=1), p_final, reduction='none').sum(axis=1) for logit_i in logits[:-1]).tolist()
        evaluators[f'ee_confusion_metric_mean/kl_div____{scope}'] = sum(kl_div_confusion) / len(kl_div_confusion)
        # if scope == 'test':

        return loss_overall, evaluators, kl_div_confusion
    
        
    
    
    def run_val(self, x_true, y_true, evaluators):
        # TODO: rozkład wyjśc, rozkład wyjść dla przykładów powyżej progu, rozkład wyjść do przykładów nie powyżej progu, liczba elementow poniżej progu
        # zbieraj confidence z tych które nie wyszły bo na końcu wybierzesz wyjście przy którym confidence był największy
        fg, repr_i = self.backbone.forward_generator(x_true), None
        self.sample_outputs = [torch.Tensor() for _ in range(x_true.size(0))]
        self.sample_exited_at = torch.zeros(x_true.size(0), dtype=torch.int) - 1
        self.max_confidence_exits = torch.zeros(x_true.size(0), dtype=torch.int) - 1
        self.max_confidences = torch.zeros(x_true.size(0)) - 1
        
        #TODO: zamień indeks heada na indeks warstwy
        self.counter_of_exits = None
        self.counter_of_exits_above_thr = None
        self.counter_of_exits_not_above_thr = None
        head_idx = 0
        
        for i in range(self.nb_of_exits):
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            if i not in self.ic_idxs: continue
            logit_i = self.internal_classifiers[head_idx](repr_i) # head_output
            
            exit_mask_local = self.find_exit(logit_i, head_idx, is_last=False)
            head_idx += 1
            # continue only if there are unresolved samples
            if (exit_mask_local).all():
                break
            # continue only with remaining sample subset
            repr_i = repr_i[~exit_mask_local]

        if not (exit_mask_local).all():
            repr_last, _ = fg.send(repr_i)
            _, output_main_head = fg.send(repr_last)
            _ = self.find_exit(output_main_head, head_idx, is_last=True)

        outputs = torch.stack(self.sample_outputs).to(self.device)
        losses = self.criterion_ic(outputs, y_true)
        # acc = acc_metric(outputs, y_true)

        evaluators[f'internal_classifier_loss/best_overall_loss____{"test"}'] = losses[1]['loss']
        evaluators[f'internal_classifier_acc/best_overall_acc____{"test"}'] = losses[1]['acc']

        return evaluators, self.counter_of_exits, self.counter_of_exits_above_thr, self.counter_of_exits_not_above_thr
    
    
    def find_exit(self, logit_i, head_idx, is_last):
        '''
        exit_mask_global (B): maska globalna, która mówi, które próbki już wyszły
        exit_mask_local (B_pozostałe): maska lokalna, która mówi, które próbki wyszły w tym kroku
        exit_mask_global_confidence: maska globalna, która mówi, które próbki nie wyszły w tym kroku, z tych które nie wyszły wcześniej
        '''
        p_i = F.softmax(logit_i, dim=1)
        head_confidences_i = p_i.max(dim=-1)[0] if self.prob_conf else self.entropy_rate(p_i)

        # na której pozycji w batchu nie ma jeszcze wyjścia
        unresolved_samples_mask = self.sample_exited_at == -1
        exit_mask_global = unresolved_samples_mask.clone()
        exit_mask_global_confidence = unresolved_samples_mask.clone()
        exit_mask_local = (head_confidences_i >= self.confidence_threshold).cpu().detach().squeeze(dim=-1)
        
        # Obsługa przypadku, gdy exit_mask_local jest skalarem
        if exit_mask_local.ndim == 0:
            exit_mask_local = torch.tensor([exit_mask_local.item()], device=logit_i.device)
            
        # wskazuje pozostałe próbki które aktualnie nie wychodzą
        exit_mask_global_confidence[unresolved_samples_mask] = ~exit_mask_local
        head_confidences_i_lower = head_confidences_i[~exit_mask_local]
        logit_i_lower = logit_i[~exit_mask_local]
        # j-ta aktualnie przetwarzana próbka która nie wyszła, k-ta pozycja w batchu
        for j, k in enumerate(exit_mask_global_confidence.nonzero().view(-1).tolist()):
            if head_confidences_i_lower[j] > self.max_confidences[k]:
                self.max_confidences[k] = head_confidences_i_lower[j]
                self.max_confidence_exits[k] = head_idx
                self.sample_outputs[k] = logit_i_lower[j]
            
        # Aktualizacja globalnych indeksów wyjścia i zapisywanie logitów dla zaklasyfikowanych próbek
        # if not is_last:
        # w masce globalnej wskazuje na pozycje które wyszły
        exit_mask_global[unresolved_samples_mask] = exit_mask_local#.to(exit_mask_global.device)
        self.sample_exited_at[exit_mask_global] = head_idx  # czy to ma być pozycja warstwy czy indeks heada?
        exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
        exit_indices_local = exit_mask_local.nonzero().view(-1).tolist()
        assert len(exit_indices_global) == len(exit_indices_local), \
            f'exit_indices_global: {exit_indices_global} exit_indices_local: {exit_indices_local}'
        for j, k in zip(exit_indices_global, exit_indices_local):
            self.sample_outputs[j] = logit_i[k]
        # else:
        if is_last:
            # logging(f"devices: {self.sample_exited_at.device.type}, {self.max_confidence_exits.device.type}, {unresolved_samples_mask.device.type}")
            unresolved_samples_mask = self.sample_exited_at == -1
            self.sample_exited_at[unresolved_samples_mask] = self.max_confidence_exits[unresolved_samples_mask]            
            
            self.counter_of_exits = dict(Counter(self.sample_exited_at.numpy()))
            self.counter_of_exits_above_thr = dict(Counter(self.sample_exited_at[unresolved_samples_mask].numpy()))
            self.counter_of_exits_not_above_thr = dict(Counter(self.sample_exited_at[~unresolved_samples_mask].numpy()))
            # exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            # for j, k in enumerate(exit_indices_global):
            #     self.sample_outputs[k] = logit_i[j]

            # # jako -1 zostają te które nie wyszły w ostatnim przez confidence
            # exit_mask_global[unresolved_samples_mask] = exit_mask_local
            # self.sample_exited_at[exit_mask_global] = head_idx

        return exit_mask_local
    
    
    def entropy_rate(self, p):
        return 1 + torch.sum(p * torch.log(p), dim=1) / np.log(p.size(-1))
    
    def plot(self, evaluators, prefix, postfix, logger):
        evaluators = {k:v for k,v in evaluators.items() if 'internal_classifier_acc' in k and 'overall' not in k and 'best' not in k}
        plot_name = f'{prefix}_plots/{postfix}'
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        
        # Tworzenie wykresu
        axs.plot(list(range(len(evaluators))), list(evaluators.values()), "o-")

        # Dodawanie tytułu i etykiet osi
        axs.set_title("Train Accuracy Across Layers")  # Dodaj tytuł wykresu
        axs.set_xlabel("Layer")  # Dodaj etykietę dla osi X
        axs.set_ylabel("Train Accuracy")  # Dodaj etykietę dla osi Y

        plot_images = {plot_name: fig}
        logger.log_plots(plot_images)

        # Opcjonalnie: Zapisz wykres jako plik PNG
        # plt.savefig(os.path.join(self.rpath, plot_name + ".png"), dpi=500)

        plt.close()
        
        
    def log_histograms(self, counters, prefix, postfix, logger, epoch):
        import wandb
        hists = {}
        for name in counters:
            plot_name = f'{prefix}_{name}/{postfix}_{epoch}'
            histogram_data = []
            for value, frequency in counters[name].items():
                histogram_data.extend([value] * frequency)

            # Logowanie histogramu
            histogram_data = [[s] for s in histogram_data]
            table = wandb.Table(data=histogram_data, columns=[name])
            hists[plot_name] = wandb.plot.histogram(table, name, title=plot_name)
            # wandb.Histogram(histogram_data)
        logger.log_histograms(hists)
