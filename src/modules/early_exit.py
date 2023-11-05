from collections import defaultdict

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
    def __init__(self, backbone, criterion, ic_idxs, confidence_threshold, sample, nb_of_exits):
        self.device = next(backbone.parameters()).device
        self.n_ics = len(ic_idxs)
        
        self.backbone = backbone
        self.criterion_ic = criterion
        self.ic_idxs = ic_idxs
        self.confidence_threshold = confidence_threshold
        self.nb_of_exits = nb_of_exits
        
        self.attach_heads(sample=sample)
        
        
        
    def attach_heads(self, sample):
        assert all(idx < self.nb_of_exits - 1 for idx in self.ic_idxs), "Możemy doczepiać heady do wszystkich prócz ostatniego wyjścia"
        fg = self._base_model.forward_generator(sample)
        input_dims = []
        
        i, x, y = 0, None, None
        while y is None:
            x, y = fg.send(x)
            if i in self.ic_idxs:
                input_dims.append(x.size(1))
            nb_of_exits += 1
        
        num_classes = y.size(-1)
        self.nb_of_exits = i - 2  # "Możemy doczepiać heady do wszystkich prócz ostatniego wyjścia"
        
        # repr_i = None
        # for i in range(self.nb_of_exits):
        #     repr_i, _ = fg.send(repr_i)
        #     if i in self.ic_idxs:
        #         input_dims.append(repr_i.size(1))
        
        # _, logit_main = fg.send(repr_i)
        # num_classes = logit_main.size(-1)
            
        self.internal_classifiers = torch.nn.ModuleList([StandardHead(in_channels, num_classes=num_classes, pool_size=4)
            for in_channels in input_dims])
        
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
    
    
    def run_train(self, x_true, y_true):
        '''
        :param x_true: data input
        :param y_true: data label
        :return:
        '''
        evaluators = defaultdict(float)
        logits = []
        fg, repr_i = self.model.forward_generator(x_true), None
        for i in range(self.ic_idxs):
            if i not in self.attach_heads: continue
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            logit_i = self.branch_classifiers[i](repr_i)
            logits.append(logit_i)

        _, logit_main = fg.send(fg.send(repr_i)[0])
        logits.append(logit_main)

        loss_overall = 0.0
        for i, logit_i in enumerate(logits):
            ce_loss_i = self.criterion_ic(logit_i, y_true)
            evaluators[f'internal_classifier_loss/{self.ic_idxs[i]}'] = ce_loss_i.item()
            evaluators[f'internal_classifier_acc/{self.ic_idxs[i]}'] = acc_metric(logit_i, y_true)
            loss_overall += 0 if ((i + 1) == len(logits) and self.is_model_frozen) else ce_loss_i

        evaluators['internal_classifier_loss/overall_loss'] = loss_overall.item()
        evaluators['internal_classifier_acc/overall_acc'] = evaluators[f'internal_classifier_acc/{self.ic_idxs[i]}']

        return loss_overall, evaluators
    
    
    def run_val(self, x_true, y_true):
        # zbieraj confidence z tych które nie wyszły bo na końcu wybierzesz wyjście przy którym confidence był największy
        fg, repr_i = self.model.forward_generator(x_true), None
        self.sample_exited_at = torch.zeros(x_true.size(0), dtype=torch.int) - 1
        self.sample_outputs = [torch.Tensor() for _ in range(x_true.size(0))]
        head_idx = 0
        for i in range(self.ic_idxs):
            if i not in self.attach_heads: continue
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            logit_i = self.branch_classifiers[i](repr_i) # head_output
            
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
        ce_loss = self.criterion_ic(outputs, y_true)
        acc = acc_metric(outputs, y_true)

        evaluators = defaultdict(float)
        evaluators['overall_loss'] = ce_loss.item()
        evaluators['overall_acc'] = acc

        return evaluators, self.sample_exited_at
    
    
    def find_exit(self, logit_i, head_idx, is_last):
        p_i = F.softmax(logit_i, dim=1)
        head_confidences_i = p_i.max(dim=-1)[0] if self.prob_conf else self.entropy_rate(p_i)

        unresolved_samples_mask = self.sample_exited_at == -1
        exit_mask_global = unresolved_samples_mask.clone()
        exit_mask_local = (head_confidences_i >= self.confidence_threshold).cpu().squeeze(dim=-1)
        
        # Obsługa przypadku, gdy exit_mask_local jest skalarem
        if exit_mask_local.ndim == 0:
            exit_mask_local = torch.tensor([exit_mask_local.item()], device=logit_i.device)
            
        # Aktualizacja globalnych indeksów wyjścia i zapisywanie logitów dla zaklasyfikowanych próbek
        if not is_last:
            exit_mask_global[unresolved_samples_mask] = exit_mask_local.to(exit_mask_global.device)
            self.sample_exited_at[exit_mask_global] = head_idx
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            exit_indices_local = exit_mask_local.nonzero().view(-1).tolist()
            assert len(exit_indices_global) == len(exit_indices_local), \
                f'exit_indices_global: {exit_indices_global} exit_indices_local: {exit_indices_local}'
            for j, k in zip(exit_indices_global, exit_indices_local):
                self.sample_outputs[j] = logit_i[k]
        else:
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            for j, k in enumerate(exit_indices_global):
                self.sample_outputs[k] = logit_i[j]

            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            self.sample_exited_at[exit_mask_global] = head_idx

        return exit_mask_local
    
    
    def entropy_rate(self, p):
        return 1 + torch.sum(p * torch.log(p), dim=1) / np.log(p.size(-1))
