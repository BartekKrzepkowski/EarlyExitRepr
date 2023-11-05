import math
from copy import deepcopy

import numpy as np
import torch

from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
from src.utils import prepare

def acc_metric(y_pred, y_true):
    correct = (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
    acc = correct / y_pred.size(0)
    return acc


def prepare_evaluators(y_pred, y_true, loss):
    acc = acc_metric(y_pred, y_true)
    evaluators = {'loss': loss.item(), 'acc': acc}
    return evaluators


class RunStats(torch.nn.Module):
    def __init__(self, model, optim):
        super().__init__()
        self.model_zero = deepcopy(model)
        self.last_model = deepcopy(model)
        self.last_model_step = deepcopy(model)
        self.model = model
        self.optim = optim
        self.model_trajectory_length_group = {k: 0.0 for k, _ in self.model.named_parameters() if _.requires_grad}
        self.model_trajectory_length_overall = 0.0
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        
    def update_checkpoint(self, global_step):
        self.last_model = deepcopy(self.model)
        if global_step % 1000 == 0:
            self.last_model_step = deepcopy(self.model)
        
    def forward(self, evaluators, distance_type):
        # self.last_model = deepcopy(self.model)
        # if global_step % 1000 == 0:
        #     self.last_model_step = deepcopy(self.model)
        
        self.model.eval()
        self.count_dead_neurons(evaluators)
        self.model_trajectory_length(evaluators)
        self.distance_between_models(self.model, self.model_zero, evaluators, distance_type, dist_label='distance_from initialization')
        self.distance_between_models(self.model, self.last_model, evaluators, distance_type, dist_label='distance_from_last_model')
        self.distance_between_models(self.model, self.last_model_step, evaluators, distance_type, dist_label='distance_from_last_model_step')
        evaluators['run_stats/excessive_length_overall'] = evaluators['run_stats/model_trajectory_length_overall'] - evaluators[f'run_stats/distance_from initialization_{distance_type}']
        self.model.train()
        return evaluators
    
    def model_trajectory_length(self, evaluators, norm_type=2.0): # odłączyć liczenie normy gradientu od liczenia długości trajektorii
        '''
        Evaluates the model trajectory length.
        '''
        lr = self.optim.param_groups[-1]['lr']
        named_parameters = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad and n in self.allowed_parameter_names]
        grad_norm_per_layer = []
        weight_norm_per_layer = []
        for n, p in named_parameters:
            weight_norm_per = torch.norm(p.data, norm_type)
            evaluators[f'run_stats_model_weight_norm_squared/{n}'] = weight_norm_per.item() ** 2
            grad_norm_per = torch.norm(p.grad, norm_type)
            evaluators[f'run_stats_model_gradient_norm_squared/{n}'] = grad_norm_per.item() ** 2
            evaluators[f'run_stats_model_grad_weight_norm_ratio_squared/{n}'] = evaluators[f'run_stats_model_gradient_norm_squared/{n}'] / (1e-9 + evaluators[f'run_stats_model_weight_norm_squared/{n}'])
            if n in self.allowed_parameter_names:
                weight_norm_per_layer.append(weight_norm_per)
                grad_norm_per_layer.append(grad_norm_per)
            self.model_trajectory_length_group[n] += lr * grad_norm_per.item()
            evaluators[f'run_stats_model_trajectory_length_group/{n}'] = self.model_trajectory_length_group[n]
            
        weight_norm = torch.norm(torch.stack(weight_norm_per_layer), norm_type).item()
        evaluators[f'run_stats/model_weight_norm_squared_overall'] = weight_norm ** 2
        grad_norm = torch.norm(torch.stack(grad_norm_per_layer), norm_type).item()
        evaluators[f'run_stats/model_gradient_norm_squared_overall'] = grad_norm ** 2
        evaluators[f'run_stats/model_grad_weight_norm_ratio_squared_overall'] = evaluators[f'run_stats/model_gradient_norm_squared_overall'] / (1e-9 + evaluators[f'run_stats/model_weight_norm_squared_overall'])
        self.model_trajectory_length_overall += lr * grad_norm
        evaluators['run_stats/model_trajectory_length_overall'] = self.model_trajectory_length_overall
    
    def distance_between_models(self, model1, model2, evaluators, distance_type, dist_label):
        def distance_between_models_l2(named_parameters1, named_parameters2, dist_label, norm_type=2.0):
            """
            Returns the l2 distance between two models.
            """
            distances = []
            for (n1, p1), (_, p2) in zip(named_parameters1, named_parameters2):
                dist = torch.norm(p1-p2, norm_type)
                if n1 in self.allowed_parameter_names:
                    distances.append(dist)
                evaluators[f'run_stats_{dist_label}_l2/{n1}'] = dist.item()
            distance = torch.norm(torch.stack(distances), norm_type)
            evaluators[f'run_stats/{dist_label}_l2'] = distance.item()
        
        def distance_between_models_cosine(named_parameters1, named_parameters2):
            """
            Returns the cosine distance between two models.
            """
            distances = []
            for (n1, p1), (_, p2) in zip(named_parameters1, named_parameters2):
                1 / 0
                distance += 1 - torch.cosine_similarity(p1.flatten(), p2.flatten())
            return distance.item()

        """
        Returns the distance between two models.
        """
        named_parameters1 = [(n, p) for n, p in model1.named_parameters() if p.requires_grad]
        named_parameters2 = [(n, p) for n, p in model2.named_parameters() if p.requires_grad]
        if distance_type == 'l2':
            distance_between_models_l2(named_parameters1, named_parameters2, dist_label=dist_label)
        elif distance_type == 'cosine':
            distance_between_models_cosine(named_parameters1, named_parameters2)
        else:
            raise ValueError(f'Distance type {distance_type} not supported.')
        
        
    def count_dead_neurons(self, evaluators):
        dead_neurons_overall = 0
        all_neurons = 0

        # Iterate over the model's modules (layers)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    dead_neurons = torch.sum(torch.all(module.weight.abs() < 1e-10, dim=1)).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/{name}'] = dead_neurons / neurons
                    dead_neurons_overall += dead_neurons
                    all_neurons += neurons
                    
            if isinstance(module, torch.nn.Conv2d):
                # Check if the layer has parameters
                if module.weight is not None:
                    # Count the number of neurons with all zero weights
                    cond = torch.all(module.weight.abs() < 1e-10, dim=1).all(dim=1).all(dim=1)
                    dead_neurons = torch.sum(cond).item()
                    neurons = module.weight.shape[0]
                    evaluators[f'run_stats_dead_neurons/{name}'] = dead_neurons / neurons
                    dead_neurons_overall += dead_neurons
                    all_neurons += neurons

        evaluators['run_stats/dead_neurons_overall'] = dead_neurons_overall / all_neurons




        

import torch
from torch.func import functional_call, vmap, grad
from sklearn.cluster import SpectralClustering 
    
class PerSampleGrad(torch.nn.Module):
    # compute loss and grad per sample 
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(device=next(model.parameters()).device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=True), in_dims=(None, None, 0, 0))
        self.allowed_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        
    def compute_loss(self, params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        predictions = functional_call(self.model, (params, buffers), (batch,))
        loss = self.criterion(predictions, targets)
        return loss, predictions

    def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
        self.model.eval()
        num_classes = y_true1.max() + 1
        matrices = {
            'similarity_11': {},
            'graham_11': {},
            'cov_11': {},
        }
        scalars = {
            'trace_of_cov_11': {},
            # 'rank_of_gradients_11': {},
            # 'rank_of_similarity_11': {},
            # # 'rank_of_weights': {},
            # 'cumm_gradients_rank_11': {},
            # 'gradients_subspace_dim_11': {},
        }
        prediction_stats = {}
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.allowed_parameter_names}
        # buffers = {k: v.detach() for k, v in self.model.named_buffers()}
        # print(list(buffers.keys()))
        buffers = {}
        
        # scalars['cumm_max_rank_of_weights'] = {k: min(v.shape)  for k, v in params.items() if 'weight' in k}
        # scalars['rank_of_weights'] = {k: self.matrix_rank(v) / min(v.shape)  for k, v in params.items() if 'weight' in k}
        # scalars['rank_of_weights']['concatenated_weights'] = sum(scalars['rank_of_weights'][tag] * scalars['cumm_max_rank_of_weights'][tag] for tag in scalars['rank_of_weights']) / sum(scalars[f'cumm_max_rank_of_weights'].values())
        # del scalars['cumm_max_rank_of_weights']
            
        ft_per_sample_grads1, y_pred1 = self.ft_criterion(params, buffers, x_true1, y_true1)
        y_pred_label1 = torch.argmax(y_pred1.data.squeeze(), dim=1)
        ft_per_sample_grads1 = {k1: v.detach().data for k1, v in ft_per_sample_grads1.items()}
        concatenated_weights1 = torch.empty((x_true1.shape[0], 0), device=x_true1.device)
        if x_true2 is not None:
            matrices.update({
                    'similarity_22': {},
                    'graham_22': {},
                    'cov_22': {},
                    'similarity_12': {},
                    'graham_12': {},
                    'cov_12': {},
                })
            scalars.update({
                    'trace_of_cov_22': {},
                    # 'rank_of_gradients_22': {},
                    # 'rank_of_similarity_22': {},
                    # 'rank_of_similarity_12': {},
                    # 'cumm_gradients_rank_22': {},
                    # 'gradients_subspace_dim_22': {},
                })
            ft_per_sample_grads2, y_pred2 = self.ft_criterion(params, buffers, x_true2, y_true2)
            y_pred_label2 = torch.argmax(y_pred2.data.squeeze(), dim=1)
            ft_per_sample_grads2 = {k2: v.detach().data for k2, v in ft_per_sample_grads2.items()}
            concatenated_weights2 = torch.empty((x_true2.shape[0], 0), device=x_true2.device)
        
        for idx in range(num_classes):
            idxs_mask1 = y_true1 == idx
            prediction_stats[f'misclassification_1_{idx}'] = (y_pred_label1[idxs_mask1] != y_true1[idxs_mask1]).float().mean().item()
            if x_true2 is not None:
                y_prob1 = torch.nn.functional.softmax(y_pred1.data.squeeze(), dim=1)
                y_prob2 = torch.nn.functional.softmax(y_pred2.data.squeeze(), dim=1)
                prediction_stats[f'misclassification_2_{idx}'] = (y_pred_label2[idxs_mask1] != y_true2[idxs_mask1]).float().mean().item()
                prediction_stats[f'mean_prob_discrepancy_{idx}'] = (y_prob1[idxs_mask1][:, idx]  - y_prob2[idxs_mask1][:, idx]).float().mean().item() 
        
        for k in ft_per_sample_grads1:
            normed_ft_per_sample_grad1, concatenated_weights1 = self.prepare_variables(ft_per_sample_grads1, concatenated_weights1, scalars, tag=k, ind='11')
            if x_true2 is not None:
                normed_ft_per_sample_grad2, concatenated_weights2 = self.prepare_variables(ft_per_sample_grads2, concatenated_weights2, scalars, tag=k, ind='22')
                self.gather_metrics(ft_per_sample_grads2, ft_per_sample_grads2, normed_ft_per_sample_grad2, normed_ft_per_sample_grad2, matrices, scalars, tag=k, hermitian=True, ind='22')
                self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads2, normed_ft_per_sample_grad1, normed_ft_per_sample_grad2, matrices, scalars, tag=k, hermitian=False, ind='12')
            
            self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads1, normed_ft_per_sample_grad1, normed_ft_per_sample_grad1, matrices, scalars, tag=k, hermitian=True, ind='11')
            
        normed_concatenated_weights1 = self.prepare_concatenated_weights(ft_per_sample_grads1, concatenated_weights1, scalars, ind='11')
        if x_true2 is not None:
            normed_concatenated_weights2 = self.prepare_concatenated_weights(ft_per_sample_grads2, concatenated_weights2, scalars, ind='22')
            self.gather_metrics(ft_per_sample_grads2, ft_per_sample_grads2, normed_concatenated_weights2, normed_concatenated_weights2, matrices, scalars, tag='concatenated_weights', hermitian=True, ind='22')
            self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads2, normed_concatenated_weights1, normed_concatenated_weights2, matrices, scalars, tag='concatenated_weights', hermitian=False, ind='12')
        
        self.gather_metrics(ft_per_sample_grads1, ft_per_sample_grads1, normed_concatenated_weights1, normed_concatenated_weights1, matrices, scalars, tag='concatenated_weights', hermitian=True, ind='11')
        # del scalars['cumm_gradients_rank_11']
        # if x_true2 is not None:
        #     del scalars['cumm_gradients_rank_22']
        self.model.train()
        return matrices, scalars, prediction_stats
    
    def trace_of_cov(self, g):
        g_mean = torch.mean(g, dim=0, keepdim=True)
        g -= g_mean
        tr = torch.mean(g.norm(dim=1)**2)
        return tr.item()
    
    # def matrix_rank(self, g, hermitian=False):
    #     pass
        # rank = np.linalg.matrix_rank(g.detach().data.cpu().numpy(), hermitian=hermitian).astype(float).mean()
        # return rank
        # return torch.linalg.matrix_rank(g, hermitian=hermitian).float().mean().item()
    
    def prepare_variables(self, ft_per_sample_grads, concatenated_weights, scalars, tag, ind: str = None):
        # if 'weight' in tag:
        #     scalars[f'cumm_gradients_rank_{ind}'][tag] = min(ft_per_sample_grads[tag].shape[1:])
        #     scalars[f'rank_of_gradients_{ind}'][tag] = self.matrix_rank(ft_per_sample_grads[tag]) / min(ft_per_sample_grads[tag].shape[1:]) # ???
        ft_per_sample_grads[tag] = ft_per_sample_grads[tag].reshape(ft_per_sample_grads[tag].shape[0], -1)
        normed_ft_per_sample_grad = ft_per_sample_grads[tag] / (1e-9 + torch.norm(ft_per_sample_grads[tag], dim=1, keepdim=True))
        concatenated_weights = torch.cat((concatenated_weights, ft_per_sample_grads[tag]), dim=1)
        # scalars[f'trace_of_cov_{ind}'][tag] = self.trace_of_cov(ft_per_sample_grads[tag])
        return normed_ft_per_sample_grad, concatenated_weights
    
    def prepare_concatenated_weights(self, ft_per_sample_grads, concatenated_weights, scalars, ind: str = None):
        # scalars[f'rank_of_gradients_{ind}']['concatenated_weights'] = sum(scalars[f'rank_of_gradients_{ind}'][tag] * scalars[f'cumm_gradients_rank_{ind}'][tag] for tag in scalars[f'rank_of_gradients_{ind}']) / sum(scalars[f'cumm_gradients_rank_{ind}'].values())
        ft_per_sample_grads['concatenated_weights'] = concatenated_weights
        normed_concatenated_weights = ft_per_sample_grads['concatenated_weights'] / (1e-9 + torch.norm(ft_per_sample_grads['concatenated_weights'], dim=1, keepdim=True))
        scalars[f'trace_of_cov_{ind}']['concatenated_weights'] = self.trace_of_cov(ft_per_sample_grads['concatenated_weights'])
        # scalars[f'gradients_subspace_dim_{ind}']['concatenated_weights'] = self.matrix_rank(normed_concatenated_weights)
        return normed_concatenated_weights
    
    def gather_metrics(self, ft_per_sample_grads1, ft_per_sample_grads2, normed_ft_per_sample_grad1, normed_ft_per_sample_grad2, matrices, scalars, tag, hermitian=False, ind: str = None):
        matrices[f'similarity_{ind}'][tag] = normed_ft_per_sample_grad1 @ normed_ft_per_sample_grad2.T
        matrices[f'graham_{ind}'][tag] = ft_per_sample_grads1[tag] @ ft_per_sample_grads2[tag].T / matrices[f'similarity_{ind}'][tag].shape[0]
        matrices[f'cov_{ind}'][tag] = (ft_per_sample_grads1[tag] - ft_per_sample_grads1[tag].mean(dim=0, keepdim=True)) @ (ft_per_sample_grads2[tag] - ft_per_sample_grads2[tag].mean(dim=0, keepdim=True)).T / matrices[f'similarity_{ind}'][tag].shape[0]
        # scalars[f'rank_of_similarity_{ind}'][tag] = self.matrix_rank(matrices[f'similarity_{ind}'][tag], hermitian=hermitian)
        if ind != '12':
            matrices[f'similarity_{ind}'][tag] = (matrices[f'similarity_{ind}'][tag] + matrices[f'similarity_{ind}'][tag].T) / 2
            matrices[f'graham_{ind}'][tag] = (matrices[f'graham_{ind}'][tag] + matrices[f'graham_{ind}'][tag].T) / 2
            matrices[f'cov_{ind}'][tag] = (matrices[f'cov_{ind}'][tag] + matrices[f'cov_{ind}'][tag].T) / 2
    
            
# wyliczyć sharpness dla macierzy podobieństwa, loader składa się z 500 przykładów
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
class Stiffness(torch.nn.Module):
    # add option to compute loss directly
    # add option with train-val
    def __init__(self, model, num_classes, x_true1, y_true1, logger=None, every_nb=1, x_true2=None, y_true2=None):
        super().__init__()
        self.per_sample_grad = PerSampleGrad(model)
        self.num_classes = num_classes
        idxs = torch.arange(0, x_true1.shape[0], every_nb)
        self.x_true1 = x_true1[idxs]
        self.y_true1 = y_true1[idxs]
        self.x_true2 = x_true2[idxs] if x_true2 is not None else None
        self.y_true2 = y_true2[idxs] if y_true2 is not None else None
        self.logger = logger
        
    def log_stiffness(self, step):
        stifness_heatmaps = {}
        stiffness_logs = {}
        stiffness_hists = {}
        matrices, scalars, prediction_stats = self.forward(self.x_true1, self.y_true1, self.x_true2, self.y_true2)
        
        # for tag in matrices[0]:
        #     # stiffness_logs[f'stiffness_sharpness/{tag}'] = self.sharpness(matrices[0][tag]['concatenated_weights'])
        #     # fig, ax  = plt.subplots(1, 1, figsize=(10, 10))
        #     # set a heatmap pallette to red-white-blue
        #     # stifness_heatmaps[f'stiffness/{tag}'] = sns.heatmap(matrices[0][tag]['concatenated_weights'].data.cpu().numpy(), ax=ax, center=0, cmap='PRGn').get_figure()
        #     # plt.close(fig)
        #     if 'similarity' in tag and '12' not in tag:
        #         labels_true = self.y_true1.cpu().numpy() if '11' in tag else self.y_true2.cpu().numpy()
        #         labels_pred, unsolicited_ratio = self.clustering(matrices[0][tag]['concatenated_weights'], labels_true)
        #         acc = (labels_pred == labels_true).sum() / labels_true.shape[0]
        #         stiffness_logs[f'clustering/accuracy_{tag}'] = acc
        #         stiffness_logs[f'clustering/unsolicited_ratio_{tag}'] = unsolicited_ratio
        #         # stiffness_hists[f'clustering/histogram_{tag}'] = labels_pred  
        
        # for tag in matrices[1]:
            # fig, ax  = plt.subplots(1, 1, figsize=(10, 10))
            # stifness_heatmaps[f'class_stiffness/{tag}'] = sns.heatmap(matrices[1][tag]['concatenated_weights'].data.cpu().numpy(), ax=ax, center=0, cmap='PRGn').get_figure()
            # plt.close(fig)
                   
        for tag in scalars[0]:
            stiffness_logs[f'traces of covs & ranks/{tag}'] = scalars[0][tag]['concatenated_weights']
            
        for tag in scalars[1]:
            stiffness_logs[f'stiffness/{tag}'] = scalars[1][tag]['concatenated_weights']
            
        for tag in prediction_stats:
            stiffness_logs[f'prediction_stats/{tag}'] = prediction_stats[tag]
        
        stiffness_logs['steps/stiffness_train'] = step
        
        # self.logger.log_figures(stifness_heatmaps, step)
        self.logger.log_scalars(stiffness_logs, step)
        # self.logger.log_histogram(stiffness_hists, step)
        
        
    def forward(self, x_true1, y_true1, x_true2=None, y_true2=None):
        matrices = defaultdict(dict)
        scalars = defaultdict(dict)
        matrices[0], scalars[0], prediction_stats = self.per_sample_grad(x_true1, y_true1, x_true2, y_true2) # [<g_i/|g_i|, g_j/|g_j|>]_{i,j}, [<g_i, g_j>]_{i,j}, [<g_i-g, g_j-g>]_{i,j}
        scalars[1]['expected_stiffness_cosine_11'] = self.cosine_stiffness(matrices[0]['similarity_11']) 
        scalars[1]['expected_stiffness_sign_11'] = self.sign_stiffness(matrices[0]['similarity_11']) 
        matrices[1]['c_stiffness_cosine_11'], scalars[1]['stiffness_between_classes_cosine_11'], scalars[1]['stiffness_within_classes_cosine_11']  = self.class_stiffness(matrices[0]['similarity_11'], y_true1, whether_sign=False)
        matrices[1]['c_stiffness_sign_11'], scalars[1]['stiffness_between_classes_sign_11'], scalars[1]['stiffness_within_classes_sign_11']  = self.class_stiffness(matrices[0]['similarity_11'], y_true1, whether_sign=True)
        if x_true2 is not None:
            scalars[1]['expected_stiffness_cosine_12'] = self.cosine_stiffness(matrices[0]['similarity_12']) 
            scalars[1]['expected_stiffness_sign_12'] = self.sign_stiffness(matrices[0]['similarity_12']) 
            scalars[1]['expected_stiffness_cosine_22'] = self.cosine_stiffness(matrices[0]['similarity_22']) 
            scalars[1]['expected_stiffness_sign_22'] = self.sign_stiffness(matrices[0]['similarity_22']) 
            matrices[1]['c_stiffness_cosine_12'], scalars[1]['stiffness_between_classes_cosine_12'], scalars[1]['stiffness_within_classes_cosine_12']  = self.class_stiffness(matrices[0]['similarity_12'], y_true1, whether_sign=False)
            matrices[1]['c_stiffness_sign_12'], scalars[1]['stiffness_between_classes_sign_12'], scalars[1]['stiffness_within_classes_sign_12']  = self.class_stiffness(matrices[0]['similarity_12'], y_true1, whether_sign=True)
            matrices[1]['c_stiffness_cosine_22'], scalars[1]['stiffness_between_classes_cosine_22'], scalars[1]['stiffness_within_classes_cosine_22']  = self.class_stiffness(matrices[0]['similarity_22'], y_true1, whether_sign=False)
            matrices[1]['c_stiffness_sign_22'], scalars[1]['stiffness_between_classes_sign_22'], scalars[1]['stiffness_within_classes_sign_22']  = self.class_stiffness(matrices[0]['similarity_22'], y_true1, whether_sign=True)
            scalars[1]['expected_stiffness_diagonal_cosine_12'], scalars[1]['expected_stiffness_diagonal_sign_12'] = self.diagonal_instance_stiffness(matrices[0]['similarity_12'])
        return matrices, scalars, prediction_stats
    
    def cosine_stiffness(self, similarity_matrices):
        expected_stiffness = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        return expected_stiffness
    
    def sign_stiffness(self, similarity_matrices):
        expected_stiffness = {k: ((torch.sum(torch.sign(v)) - torch.diagonal(torch.sign(v)).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in similarity_matrices.items()}
        return expected_stiffness
    
    def class_stiffness(self, similarity_matrices, y_true, whether_sign=False):
        c_stiffness = {}
        # extract the indices into dictionary from y_true tensor where the class is the same
        indices = {i: torch.where(y_true == i)[0] for i in range(self.num_classes)}
        indices = {k: v for k, v in indices.items() if v.shape[0] > 0}
        for k, similarity_matrix in similarity_matrices.items():
            c_stiffness[k] = torch.zeros((self.num_classes, self.num_classes), device=y_true.device)
            for c1, idxs1 in indices.items():
                for c2, idxs2 in indices.items():
                    sub_matrix = similarity_matrix[idxs1, :][:, idxs2]
                    sub_matrix = torch.sign(sub_matrix) if whether_sign else sub_matrix
                    c_stiffness[k][c1, c2] = torch.mean(sub_matrix) if c1 != c2 else (torch.sum(sub_matrix) - sub_matrix.size(0)) / (sub_matrix.size(0)**2 - sub_matrix.size(0))
                    
        stiffness_between_classes = {k: ((torch.sum(v) - torch.diagonal(v).sum()) / (v.size(0)**2 - v.size(0))).item() for k, v in c_stiffness.items()}
        stiffness_within_classes = {k: (torch.diagonal(v).sum() / v.size(0)).item() for k, v in c_stiffness.items()}
        
        return c_stiffness, stiffness_between_classes, stiffness_within_classes  
    
    def diagonal_instance_stiffness(self, similarity_matrices):
        expected_diag_stiffness_cosine = {k: torch.mean(torch.diag(v)).item() for k, v in similarity_matrices.items()}
        expected_diag_stiffness_sign = {k: torch.mean(torch.sign(torch.diag(v))).item() for k, v in similarity_matrices.items()}
        return expected_diag_stiffness_cosine, expected_diag_stiffness_sign
        
    def sharpness(self, similarity_matrix):
        w, _ = torch.linalg.eig(similarity_matrix)
        max_eig = torch.max(w.real) # .abs()??
        return max_eig.item()
    
    def clustering(self, similarity_matrix, labels_true):
        similarity_matrix_ = similarity_matrix.cpu().numpy()
        labels_pred = SpectralClustering(n_clusters=self.num_classes, affinity='precomputed', n_init=100, assign_labels='discretize').fit_predict((1+similarity_matrix_)/2)
        labels_pred, unsolicited_ratio = self.retrieve_info(labels_pred, labels_true)
        return labels_pred, unsolicited_ratio
    
    
    def retrieve_info(self, cluster_labels, y_train):
        ## ValueError: attempt to get argmax of an empty sequence: dist.argmax()
        # Initializing
        unsolicited_ratio = 0.0
        denominator = 0.0
        reference_labels = {}
        # For loop to run through each label of cluster label
        for label in range(len(np.unique(y_train))):
            index = np.where(cluster_labels==label, 1, 0)
            dist = np.bincount(y_train[index==1])
            num = dist.argmax()
            unsolicited_ratio += (dist.sum() - dist.max())
            denominator += dist.sum()
            reference_labels[label] = num
        proper_labels = [reference_labels[label] for label in cluster_labels]
        proper_labels = np.array(proper_labels)
        unsolicited_ratio /= denominator
        return proper_labels, unsolicited_ratio

    