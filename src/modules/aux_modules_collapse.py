from collections import defaultdict

import numpy as np
from scipy.linalg import eigh
import torch


def variance_angle_pairwise(x_data, y_true, evaluators): # zamiast cos do średniej, rozważ między wszystkimi możliwymi wektorami
    eps = torch.finfo(torch.float32).eps
    classes = np.unique(y_true.cpu().numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        for k in range(len(internal_repr)):
            internal_repr[k] /= (torch.norm(internal_repr[k]) + eps)
            
        cos = internal_repr @ internal_repr.T
        abs_angle_matrix = torch.abs(torch.acos(cos))
        
        general_mean = torch.triu(abs_angle_matrix, diagonal=1)
        general_mean = general_mean[general_mean.nonzero(as_tuple=True)].flatten().mean()
        
        for i in range(denom_class):
            idxs_i = np.where(y_true.cpu().numpy() == i)[0]
            sub_matrix = abs_angle_matrix[idxs_i][:, idxs_i]
            sub_matrix = torch.triu(sub_matrix, diagonal=1)
            sub_vector = sub_matrix[sub_matrix.nonzero(as_tuple=True)].flatten()
            off_diag_mean = sub_vector.mean()
            within_class_cov += (sub_vector - off_diag_mean).pow(2).mean().item()
            between_class_cov += (off_diag_mean - general_mean).pow(2).item()

        within_class_cov /= denom_class
        between_class_cov /= denom_class
        total_class_cov = within_class_cov + between_class_cov
        
        evaluators[f'within_cov_abs_angle/{name}'] = within_class_cov
        evaluators[f'between_cov_abs_angle/{name}'] = between_class_cov
        evaluators[f'total_cov_abs_angle/{name}'] = total_class_cov
        
        normalized_wcc = within_class_cov / (total_class_cov + eps) 
        normalized_bcc = between_class_cov / (total_class_cov + eps)
        
        
        evaluators[f'within_cov_abs_angle_normalized/{name}'] = normalized_wcc
        evaluators[f'between_cov_abs_angle_normalized/{name}'] = normalized_bcc
    return evaluators


def variance_eucl(x_data, y_true, evaluators):
    eps = torch.finfo(torch.float32).eps
    classes = np.unique(y_true.cpu().numpy())
    denom_class = classes.shape[0]
    for name, internal_repr in x_data.items():
        within_class_cov = 0.0
        between_class_cov = 0.0
        general_mean = torch.mean(internal_repr, dim=0, keepdim=True).detach()  # (1, D)
        for c in classes:
            class_internal_repr = internal_repr[y_true == c].detach()  # (N_c, D)
            class_mean = torch.mean(class_internal_repr, dim=0, keepdim=True)  # (1, D)
            class_internal_repr_sub = class_internal_repr - class_mean  # (N_c, D)
            # print(class_internal_repr_sub.shape)
            # for sample in class_internal_repr_sub:
            #     within_class_cov += sample.unsqueeze(1) @ sample.unsqueeze(0)
            # within_class_cov = (class_internal_repr_sub.unsqueeze(2) @ class_internal_repr_sub.unsqueeze(1)).mean(dim=0)  # (N_c, D, 1) x (N_c, 1, D) -> (D, D)
            
            S = 250 # depends on GPU available (GB)
            K = class_internal_repr_sub.shape[0] // S
            within_class_cov += sum([(class_internal_repr_sub[S*i:S*(i+1)].unsqueeze(2) @ class_internal_repr_sub[S*i:S*(i+1)].unsqueeze(1)).mean(dim=0) for i in range(K)]) / K  # (N_c, D, 1) x (N_c, 1, D) -> (D, D)
            between_sample = (class_mean - general_mean)
            between_class_cov += between_sample.T @ between_sample  # (D, 1) x (1, D) -> (D, D)
        
        within_class_cov /= denom_class  # (D, D)
        between_class_cov /= denom_class  # (D, D)
        total_class_cov = within_class_cov + between_class_cov  # (D, D)
        
        trace_wcc = torch.trace(within_class_cov)
        trace_bcc = torch.trace(between_class_cov)
        trace_tcc = torch.trace(total_class_cov)
        
        evaluators[f'within_cov_eucl_(trace)/{name}'] = trace_wcc.item()
        evaluators[f'between_cov_eucl_(trace)/{name}'] = trace_bcc.item()
        evaluators[f'total_cov_eucl_(trace)/{name}'] = trace_tcc.item()
        
        normalized_wcc = trace_wcc / (trace_tcc + eps) 
        normalized_bcc = trace_bcc / (trace_tcc + eps)
        
        evaluators[f'within_cov_eucl_normalized/{name}'] = normalized_wcc.item()
        evaluators[f'between_cov_eucl_normalized/{name}'] = normalized_bcc.item()
        
        
        rank_wcc = torch.linalg.matrix_rank(within_class_cov)
        # rank_bcc = torch.linalg.matrix_rank(between_class_cov)
        # rank_tcc = torch.linalg.matrix_rank(total_class_cov)
        
        evaluators[f'within_cov_rank/{name}'] = rank_wcc.item()
        # evaluators[f'between_cov_rank/{name}'] = rank_bcc.item()
        # evaluators[f'total_cov_rank/{name}'] = rank_tcc.item()
        
        # A = within_class_cov.T @ within_class_cov
        # square_stable_rank_wcc = torch.trace(A) #/ eigh(A.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0]#torch.lobpcg(A, k=1)[0][0]
        # B = between_class_cov.T @ between_class_cov
        # square_stable_rank_bcc = torch.trace(B) #/ eigh(B.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0]#torch.lobpcg(B, k=1)[0][0]
        # C = total_class_cov.T @ total_class_cov
        # square_stable_rank_tcc = torch.trace(C) #/ eigh(C.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0]#torch.lobpcg(C, k=1)[0][0]
        
        # evaluators[f'within_cov_square_stable_rank/{name}'] = square_stable_rank_wcc.item()
        # evaluators[f'between_cov_square_stable_rank/{name}'] = square_stable_rank_bcc.item()
        # evaluators[f'total_cov_square_stable_rank/{name}'] = square_stable_rank_tcc.item()


import torch
from torch.func import functional_call, vmap, grad
from sklearn.cluster import SpectralClustering

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
    
class TunnelGrad(torch.nn.Module):
    # compute loss and grad per sample 
    def __init__(self, loaders, model, cutoff):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.loaders = loaders
        self.cutoff = cutoff
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.eps = torch.finfo(torch.float32).eps
        self.logger = None
        self.subsampling = defaultdict(lambda: None)
        
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss, predictions

    def forward(self, step, scope, phase):
        self.model.eval()
        DATASET_NAME = 'test'
        x_test = torch.cat([x for x, _ in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        y_test = torch.cat([y for _, y in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        classes = np.unique(y_test.cpu().numpy())
        prefix = lambda a, b : f'ranks_grads_{a}_{b}'
        postfix = f'____{scope}____{phase}'
        evaluators = {}
        per_class_size = 500
        chunk_size = 250
        
        
        idxs = []
        for i in range(len(classes)):
            idxs_i = np.where(y_test.cpu().numpy() == i)[0]
            sampled_idxs_i = np.random.choice(idxs_i, size=per_class_size, replace=False)
            idxs.append(sampled_idxs_i)
        
        idxs = np.concatenate(idxs)
        x_test = x_test[idxs]
        y_test = y_test[idxs]

        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names and 'bias' not in k and 'downsample' not in k}
        buffers = {}
        
        per_sample_grads, y_pred = None, None
        
        for i in range(x_test.shape[0] // chunk_size):  # accumulate grads
            per_sample_grads_, y_pred_ = self.ft_criterion(params, buffers, x_test[i*chunk_size: (i+1)*chunk_size], y_test[i*chunk_size: (i+1)*chunk_size])
            per_sample_grads_ = {k1: v.detach().data for k1, v in per_sample_grads_.items()}
            self.prepare_variables(per_sample_grads_)
            self.sample_feats(per_sample_grads_)
            per_sample_grads, y_pred = self.update(per_sample_grads, y_pred, per_sample_grads_, y_pred_)
        y_pred_label = torch.argmax(y_pred.data.squeeze(), dim=1)
        
        # concatenated_grads = torch.empty((x_test.shape[0], 0), device=x_test.device)
        
        for c in classes:
            idxs_mask = y_test == c
            evaluators[f'misclassification_per_class/{c}'] = (y_pred_label[idxs_mask] != y_test[idxs_mask]).float().mean().item()
            
        
        # self.prepare_variables(per_sample_grads, concatenated_grads)
        # self.sample_feats(per_sample_grads)
            
        evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'feature'), postfix, normalize=False, batch_first=False, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('nonnormalized', 'batch'), postfix, normalize=False, batch_first=True, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'feature'), postfix, normalize=True, batch_first=False, risky_names=('concatenated_grads'))
        # evaluators = self.prepare_and_calculate_ranks(per_sample_grads, evaluators, prefix('normalized', 'batch'), postfix, normalize=True, batch_first=True, risky_names=('concatenated_grads'))
        
        variance_angle_pairwise(per_sample_grads, y_test, evaluators)
        
        self.model.train()
        evaluators['steps/tunnel_grads'] = step
        self.logger.log_scalars(evaluators, step)
        
        
    def prepare_variables(self, per_sample_grads, concatenated_grads=None): # sampluj gdy za duże
        for weight_name in per_sample_grads:
            per_sample_grads[weight_name] = per_sample_grads[weight_name].reshape(per_sample_grads[weight_name].shape[0], -1)
            # concatenated_grads = per_sample_grads[weight_name]#torch.cat((concatenated_grads, per_sample_grads[weight_name]), dim=1)
        # per_sample_grads['concatenated_grads'] = concatenated_grads
        
    def sample_feats(self, per_sample_grads):
        for name in per_sample_grads:
            if name in self.subsampling:
                per_sample_grads[name] = self.adjust_representation(per_sample_grads[name], name)
            elif per_sample_grads[name].size(1) > self.cutoff:
                self.subsampling[name] = torch.randperm(per_sample_grads[name].size(1))[:self.cutoff].sort()[0]
                per_sample_grads[name] = self.adjust_representation(per_sample_grads[name], name)
                
    def adjust_representation(self, grads, name):
        representation = torch.index_select(grads, 1, self.subsampling[name].to(self.device))
        return representation
    
    def update(self, per_sample_grads_old, y_pred_old, per_sample_grads_new, y_pred_new):
        if per_sample_grads_old is None:
            return per_sample_grads_new, y_pred_new
        for name in per_sample_grads_old:
            per_sample_grads_old[name] = torch.cat((per_sample_grads_old[name], per_sample_grads_new[name]), dim=0)
        y_pred_old = torch.cat((y_pred_old, y_pred_new), dim=0)
        return per_sample_grads_old, y_pred_old
    
    def prepare_and_calculate_ranks(self, matrices, evaluators, prefix, postfix, normalize=False, batch_first=True, risky_names=()):
        for name, matrix in matrices.items():
            if name in risky_names and not batch_first:
                continue
            if normalize:
                matrix = matrix / (torch.norm(matrix, dim=1, keepdim=True) + self.eps)
            ranks = self.calculate_rank(matrix, batch_first)
            denom = matrix.size(0) if batch_first else matrix.T.size(0)
            
            name_dict = f'{prefix}/{name}{postfix}'
            name_dict_ratio = f'{prefix}_ratio/{name}{postfix}'
            # name_dict_null = f'{prefix}_null/{name}{postfix}'
            evaluators[name_dict] = ranks[0]
            evaluators[name_dict_ratio] = ranks[0] / denom # check if dim makes sense
            # evaluators[name_dict_null] = denom - ranks[0]
            
            # name_dict_square_stable = f'{prefix}_square_stable/{name}{postfix}'
            # name_dict_ratio_square_stable = f'{prefix}_ratio_square_stable/{name}{postfix}'
            # name_dict_null_square_stable = f'{prefix}_null_square_stable/{name}{postfix}'
            # evaluators[name_dict_square_stable] = ranks[1]
            # evaluators[name_dict_ratio_square_stable] = ranks[1] / denom # check if dim makes sense
            # evaluators[name_dict_null_square_stable] = denom - ranks[1]
        return evaluators
    
    def calculate_rank(self, matrix, batch_first): # jedyny pomysł to z paddingiem macierzy do maksymalnej
        matrix = matrix if batch_first else matrix.T
        gramian_matrix = matrix @ matrix.T
        rank = torch.linalg.matrix_rank(gramian_matrix).item()
        square_stable_rank = 0#(torch.diag(gramian_matrix).sum()).item() #/ torch.lobpcg(gramian_matrix, k=1)[0][0]).item()
        return rank, square_stable_rank
    
        