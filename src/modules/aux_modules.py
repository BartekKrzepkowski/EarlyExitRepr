from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.linalg import eigh
import torch
from torch.distributions import Categorical

from src.utils import prepare
from src.utils.utils_optim import get_every_but_forbidden_parameter_names, FORBIDDEN_LAYER_TYPES
from src.utils.utils_metrics import correct_metric
from src.modules.aux_modules_collapse import variance_eucl

from functorch import combine_state_for_ensemble
from torch.func import functional_call, vmap, grad 
class TunnelandProbing(torch.nn.Module):
    def __init__(self, loaders, model, num_classes, optim_type, optim_params,
                 reprs_hook, epochs_probing):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.loaders = loaders
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.num_classes = num_classes
        self.optim_type = optim_type
        self.optim_params = optim_params
        self.hooks_reprs = reprs_hook
        self.epochs_probing = epochs_probing
        self.logger = None
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        self.eps = torch.finfo(torch.float32).eps
        
        # self.vmap_ranks = vmap(self.calculate_rank, in_dims=(None, 0))
        
    def forward(self, step, scope, phase):
        self.model.eval()
        DATASET_NAME = 'test'
        x_test = torch.cat([x for x, _ in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        y_test = torch.cat([y for _, y in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        with torch.no_grad():
            self.model(x_test)
        internal_representations_test = self.hooks_reprs.get_assets()  # (B, D)
        self.hooks_reprs.reset()
        evaluators = {}
        
        # calculate ranks
        postfix = f'____{scope}____{phase}'
        named_weights = {n: p.reshape(p.size(0), -1) for n, p in self.model.named_parameters() if 'weight' in n and n in self.penalized_parameter_names}
        evaluators = self.prepare_and_calculate_ranks(internal_representations_test, evaluators, prefix=f'ranks_representations', postfix=postfix)
        evaluators = self.prepare_and_calculate_ranks(named_weights, evaluators, prefix=f'ranks_weights', postfix=postfix)
        
        variance_eucl(internal_representations_test, y_test, evaluators)
        
        # # prepare heads
        # input_dims = [representation.size(1) for representation in internal_representations_test]
        # output_dims = len(input_dims) * [self.num_classes]
        # self.heads = self.prepare_heads(input_dims, output_dims).to(self.device)
        # self.optim = prepare.prepare_optim_and_scheduler(self.heads, self.optim_type, deepcopy(self.optim_params))[0]
        # self.fhead, params, buffers = combine_state_for_ensemble(self.heads)
        # vmap_heads = vmap(self.fhead, in_dims=(None, None, 0))
        # vmap_heads = vmap(self.compute_loss, in_dims=(None, 0))
        
        # probe layers
        # accs_epoch = []
        # losses_epoch = []
        self.model.train()
        # for _ in range(self.epochs_probing): # ustalić warunek stopu zbieżności głowic
        #     self.train_heads(vmap_heads, params, buffers) # train
        #     # self.train_heads(vmap_heads) # train
        #     with torch.no_grad():
        #         accs_test, losses_test = self.test_heads(vmap_heads, params, buffers, internal_representations_test, y_test) # test
        #       # gather 
        #     accs_epoch.append(accs_test)
        #     losses_epoch.append(losses_test)
        # # gather the best over the epochs
        # accs_epoch = torch.tensor(accs_epoch).max(dim=0)[0].reshape(-1)
        # losses_epoch = torch.tensor(losses_epoch).min(dim=0)[0].reshape(-1)
        # # log results
        # self.prepare_evaluators(evaluators, accs_epoch, losses_epoch, scope, phase)
        evaluators['steps/tunnel'] = step
        self.logger.log_scalars(evaluators, step)
    
    def prepare_and_calculate_ranks(self, matrices, evaluators, prefix, postfix):
        # _, matrices = zip(*matrices)
        # print(matrices.device)
        # matrices = matrices.to(self.device)
        # matrices = torch.nested.nested_tensor(list(matrices))
        is_batch_first = False if 'weight' in prefix else True
        
        # ranks = self.vmap_ranks(transpose, matrices)
        for name, matrix in matrices.items():
            ranks = self.calculate_rank(is_batch_first, matrix)
            denom = matrix.T.size(0) if is_batch_first else matrix.size(0)
            
            name_dict = f'{prefix}/{name}{postfix}'
            name_dict_ratio = f'{prefix}_ratio/{name}{postfix}'
            # name_dict_null = f'{prefix}_null/{name}{postfix}'
            evaluators[name_dict] = ranks[0]
            evaluators[name_dict_ratio] = ranks[0] / denom # check if dim makes sense
            # evaluators[name_dict_null] = denom - ranks[0]
            
            # name_dict_square_stable = f'{prefix}_square_stable/{name}{postfix}'
            # name_dict_ratio_square_stable = f'{prefix}_ratio_square_stable/{name}{postfix}'
            # # name_dict_null_square_stable = f'{prefix}_null_square_stable/{name}{postfix}'
            # evaluators[name_dict_square_stable] = ranks[1]
            # evaluators[name_dict_ratio_square_stable] = ranks[1] / denom # check if dim makes sense
            # # evaluators[name_dict_null_square_stable] = denom - ranks[1]
        return evaluators

    def calculate_rank(self, is_batch_first, matrix):
        matrix = matrix.T if is_batch_first else matrix  # if batch first then transpose to get features
        gramian_matrix = matrix @ matrix.T
        rank = torch.linalg.matrix_rank(gramian_matrix).item()
        square_stable_rank = 0#torch.trace(gramian_matrix).item() #/ eigh(gramian_matrix.detach().cpu().numpy(), subset_by_index=[1, 1], eigvals_only=True)[0] #(torch.lobpcg(gramian_matrix, k=1)[0][0]).item()
        return rank, square_stable_rank
    
    # def calculate_rank_via_svd(self, transpose, matrix):
    #     matrix = matrix.T if transpose else matrix
    #     cov_matrix = matrix#torch.cov(matrix)
    #     singulars = torch.linalg.svdvals(cov_matrix)
    #     rank = (singulars > (singulars[0] * max(cov_matrix.size()) * self.eps)).sum()
    #     singulars = singulars ** 2
    #     square_stable_rank = singulars.sum() / (singulars[0] + self.eps)
    #     return rank, square_stable_rank
    
    def prepare_heads(self, input_dims, output_dims):
        heads = []
        for i in range(len(input_dims)):
            head = torch.nn.Linear(input_dims[i], output_dims[i])
            heads.append(head)
        heads = torch.nn.ModuleList(heads)
        return heads
          
    def train_heads(self, vmap_heads, params, buffers):
        self.fhead.train()
        for i, (x_train, y_train) in enumerate(self.loaders['train']):
            # if i > 20:
            #     break
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            _ = self.model(x_train)
            internal_representations = self.hooks_reprs.get_assets()
            self.hooks_reprs.reset()
            losses_train = []
            y_preds = vmap_heads(internal_representations.items(), params, buffers)
            losses_train = [self.criterion(y_pred, y_train) for y_pred in y_preds]
            # losses_train = vmap_heads(internal_representations.items(), self.heads)
            loss = sum(losses_train) / len(losses_train)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
    
    def test_heads(self, vmap_heads, params, buffers, representations_test, y_test):
        global_denom = 0.0
        accs_test = np.array([])
        losses_test = np.array([])
        self.fhead.eval()
        for reprs_batch, y_test0 in zip(torch.split(representations_test, 1000, dim=0), torch.split(y_test, 1000, dim=0)):
            denom = y_test0.size(0)
            y_preds = vmap_heads(reprs_batch.items(), params, buffers)
            losses_test += np.array([self.criterion(y_pred, y_test0) * denom for y_pred in y_preds])
            accs_test += np.array([correct_metric(y_pred, y_test0) for y_pred in y_preds])
            global_denom += denom

        accs_test = accs_test / global_denom
        losses_test = losses_test / global_denom
        return accs_test, losses_test
    
    def prepare_evaluators(self, evaluators, accs, losses, scope, phase):
        for i in range(len(accs)):
            evaluators[f'linear_probing/accuracy_{i}____{scope}____{phase}'] = accs[i].item()
            evaluators[f'linear_probing/loss_{i}____{scope}____{phase}'] = losses[i].item()
            
    # def compute_loss(self, representations, model):
    #     x_true, y_true = representations
    #     y_pred = model(x_true)


class TunnelandProbingBasic(torch.nn.Module):
    def __init__(self, loaders, model, num_classes, optim_type, optim_params,
                 reprs_hook, epochs_probing):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.loaders = loaders
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.num_classes = num_classes
        self.optim_type = optim_type
        self.optim_params = optim_params
        self.hooks_reprs = reprs_hook
        self.epochs_probing = epochs_probing
        self.logger = None
        
    def forward(self, step, checkpoint_step):
        self.model.eval()
        DATASET_NAME = 'test'
        x_test = torch.cat([x for x, _ in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        y_test = torch.cat([y for _, y in self.loaders[DATASET_NAME]], dim=0).to(self.device)
        with torch.no_grad():
            _ = self.model(x_test)
        representations_test = self.hooks_reprs.callback.activations
        named_weights = [(n, p.reshape(p.size(0), -1)) for n, p in self.model.named_parameters() if 'weight' in n]
        self.hooks_reprs.reset()
        evaluators = {}
        
        input_dims = [min(rep[1].size(-1), 8000) for rep in representations_test]
        output_dims = len(input_dims) * [self.num_classes]
        self.heads = self.prepare_heads(input_dims, output_dims).to(self.device)
        self.optim = prepare.prepare_optim_and_scheduler(self.heads, self.optim_type, deepcopy(self.optim_params))[0]
        
        evaluators = self.prepare_and_calculate_ranks(representations_test, evaluators, prefix=f'ranks_representations_{DATASET_NAME}')
            
        evaluators = self.prepare_and_calculate_ranks(named_weights, evaluators, prefix=f'ranks_weights_{DATASET_NAME}')

        
        accs_epoch = []
        losses_epoch = []
        self.model.train()
        for _ in range(self.epochs_probing): # ustalić warunek stopu zbieżności głowic
            self.heads.train()
            for i, (x_train, y_train) in enumerate(self.loaders['train']):
                if i > 10:
                    break
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                
                _ = self.model(x_train)
                reprs = self.hooks_reprs.callback.activations
                self.hooks_reprs.reset()
                losses_train = []
                for i, head in enumerate(self.heads):
                    name, rep = reprs[i]
                    if rep.size(1) > 8000:
                        rep = rep[:, self.hooks_reprs.callback.subsampling[name]]
                    output = head(rep)
                    loss = self.criterion(output, y_train)
                    losses_train.append(loss)
            
                loss = sum(losses_train) / len(losses_train)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

            accs_test = []
            losses_head = []
            self.heads.eval()
            for i, head in enumerate(self.heads):
                name, rep = representations_test[i]
                if rep.size(1) > 8000:
                    rep = rep[:, self.hooks_reprs.callback.subsampling[name]]
                output = head(rep)
                loss = self.criterion(output, y_test)
                acc = (torch.argmax(output, dim=1) == y_test).float().mean()
                losses_head.append(loss)
                accs_test.append(acc)
                
            accs_epoch.append(accs_test)
            losses_epoch.append(losses_head)
            
        accs_epoch = torch.tensor(accs_epoch).max(dim=0)[0].reshape(-1)
        losses_epoch = torch.tensor(losses_epoch).max(dim=0)[0].reshape(-1)
        self.prepare_evaluators(evaluators, accs_epoch, losses_epoch)
        
        evaluators['steps/tunnel'] = step
        self.logger.log_scalars(evaluators, step)
        
    def prepare_and_calculate_ranks(self, representations, evaluators, prefix):
        streams = [torch.cuda.Stream() for _ in representations] if self.device.type == 'cuda' else [None for _ in representations]
        transpose = False if 'weight' in prefix else True
        ranks = [self.calculate_rank(name, matrix, stream, transpose) for (name, matrix), stream in zip(representations, streams)]
        if self.device.type == 'cuda':
            [stream.synchronize() for stream in streams]
        for i, (name, matrix) in enumerate(representations):
            name_dict = f'{prefix}/{name}'
            name_dict_ratio = f'{prefix}_ratio/{name}'
            # name_dict_null = f'{prefix}_null/{name}'
            evaluators[name_dict] = ranks[i]
            evaluators[name_dict_ratio] = ranks[i] / matrix.size(1)
            # evaluators[name_dict_null] = matrix.size(1) - ranks[i]
        return evaluators


    def calculate_rank(self, name, matrix, stream, transpose):
        with torch.cuda.stream(stream):
            matrix = matrix.data.detach()
            matrix = matrix.T if transpose else matrix
            if matrix.size(0) > 8000:
                matrix = matrix[self.hooks_reprs.callback.subsampling[name]]
            cov_matrix = torch.cov(matrix)
            rank = torch.linalg.matrix_rank(cov_matrix).item()
        return rank

    def prepare_heads(self, input_dims, output_dims):
        heads = []
        for i in range(len(input_dims)):
            heads.append(torch.nn.Linear(input_dims[i], output_dims[i]))
        heads = torch.nn.ModuleList(heads)
        return heads
    
    def prepare_evaluators(self, evaluators, accs, losses):
        for i in range(len(accs)):
            evaluators[f'linear_probing/accuracy_{i}'] = accs[i].item()
            evaluators[f'linear_probing/loss_{i}'] = losses[i].item()


class TraceFIMB(torch.nn.Module):
    def __init__(self, x_held_out, model, criterion, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.x_held_out = x_held_out
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.logger = None

    def forward(self, step):
        # saved_training = self.model.training
        # self.model.eval()
        y_pred = self.model(self.x_held_out)
        # self.model.train(saved_training)
        y_sampled = Categorical(logits=y_pred).sample()
        loss = self.criterion(y_pred, y_sampled)
        params_names, params = zip(*[(n, p) for n, p in self.model.named_parameters() if p.requires_grad])
        # self.model.train(saved_training)
        grads = torch.autograd.grad(
            loss,
            params,)
            # retain_graph=True,
            # create_graph=True)
        evaluators = defaultdict(float)
        overall_trace = 0.0
        for param_name, gr in zip(params_names, grads):
            if gr is not None:
                trace_p = (gr**2).sum()
                evaluators[f'trace_fim/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p.item()
        
        evaluators['trace_fim/overall_trace'] = overall_trace
        evaluators['steps/trace_fim'] = step
        self.logger.log_scalars(evaluators, step)   


class TraceFIM(torch.nn.Module):
    def __init__(self, x_held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.x_held_out = x_held_out
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(self.grad_and_trace, in_dims=(None, None, 0), randomness="different")
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.labels = torch.arange(num_classes).to(self.device)
        self.logger = None
        
    def compute_loss(self, params, buffers, sample):
        batch = sample.unsqueeze(0)
        y_pred = functional_call(self.model, (params, buffers), (batch, ))
        # y_sampled = Categorical(logits=y_pred).sample()
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze(-1)
        loss = self.criterion(y_pred, y_sampled)
        return loss
    
    def grad_and_trace(self, params, buffers, sample):
        sample_traces = {}
        sample_grads = grad(self.compute_loss, has_aux=False)(params, buffers, sample)
        for param_name in sample_grads:
            gr = sample_grads[param_name]
            if gr is not None:
                trace_p = (torch.pow(gr, 2)).sum()
                sample_traces[param_name] = trace_p
        return sample_traces

    def forward(self, step):
        self.model.eval()
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.penalized_parameter_names and v.requires_grad}
        buffers = {}
        ft_per_sample_grads = self.ft_criterion(params, buffers, self.x_held_out)
        ft_per_sample_grads = {k: v.detach().data for k, v in ft_per_sample_grads.items()}
        evaluators = defaultdict(float)
        overall_trace = 0.0
        for param_name in ft_per_sample_grads:
            trace_p = ft_per_sample_grads[param_name].mean()
            evaluators[f'trace_fim/{param_name}'] += trace_p.item()
            if param_name in self.penalized_parameter_names:
                overall_trace += trace_p.item()
         
        evaluators[f'trace_fim/overall_trace'] = overall_trace
        evaluators['steps/trace_fim'] = step
        self.model.train()
        self.logger.log_scalars(evaluators, step)
        
        
class TraceFIMbackup(torch.nn.Module):
    def __init__(self, x_held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.x_held_out = x_held_out
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=False), in_dims=(None, None, 0), randomness="different")
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.labels = torch.arange(num_classes).to(self.device)
        self.logger = None
        
    def compute_loss(self, params, buffers, sample):
        batch = sample.unsqueeze(0)
        y_pred = functional_call(self.model, (params, buffers), (batch, ))
        # y_sampled = Categorical(logits=y_pred).sample()
        prob = torch.nn.functional.softmax(y_pred, dim=1)
        idx_sampled = prob.multinomial(1)
        y_sampled = self.labels[idx_sampled].long().squeeze(-1)
        loss = self.criterion(y_pred, y_sampled)
        return loss

    def forward(self, step):
        self.model.eval()
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.penalized_parameter_names and v.requires_grad}
        buffers = {}
        ft_per_sample_grads = self.ft_criterion(params, buffers, self.x_held_out)
        ft_per_sample_grads = {k: v.detach().data for k, v in ft_per_sample_grads.items()}
        evaluators = defaultdict(float)
        overall_trace = 0.0
        for param_name in ft_per_sample_grads:
            gr = ft_per_sample_grads[param_name]
            if gr is not None:
                trace_p = (gr**2).sum() / gr.size(0)
                evaluators[f'trace_fim/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p.item()
         
        evaluators[f'trace_fim/overall_trace'] = overall_trace
        evaluators['steps/trace_fim'] = step
        self.model.train()
        self.logger.log_scalars(evaluators, step)    


class TraceFIMverification(torch.nn.Module):
    '''Eq to TraceFIM'''
    def __init__(self, x_held_out, model, num_classes):
        super().__init__()
        self.device = next(model.parameters()).device
        self.x_held_out = x_held_out
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.ft_criterion = vmap(grad(self.compute_loss, has_aux=False), in_dims=(None, None, 0, 0))
        self.penalized_parameter_names = get_every_but_forbidden_parameter_names(self.model, FORBIDDEN_LAYER_TYPES)
        print("penalized_parameter_names: ", self.penalized_parameter_names)
        self.logger = None
        
    def compute_loss(self, params, buffers, sample, y_sampled):
        batch = sample.unsqueeze(0)
        y_sampled = y_sampled.unsqueeze(0)
        y_pred = functional_call(self.model, (params, buffers), (batch, ))    
        loss = self.criterion(y_pred, y_sampled)
        return loss

    def forward(self, step):
        self.model.eval()
        y_pred = self.model(self.x_held_out)
        y_sampled = Categorical(logits=y_pred).sample()
        params = {k: v.detach() for k, v in self.model.named_parameters() if k in self.penalized_parameter_names and v.requires_grad}
        buffers = {}
        ft_per_sample_grads = self.ft_criterion(params, buffers, self.x_held_out, y_sampled)
        ft_per_sample_grads = {k: v.detach().data for k, v in ft_per_sample_grads.items()}
        evaluators = defaultdict(float)
        overall_trace = 0.0
        for param_name in ft_per_sample_grads:
            gr = ft_per_sample_grads[param_name]
            if gr is not None:
                trace_p = (gr**2).sum() / gr.size(0)
                evaluators[f'trace_fim/{param_name}'] += trace_p.item()
                if param_name in self.penalized_parameter_names:
                    overall_trace += trace_p.item()
         
        evaluators[f'trace_fim/overall_trace'] = overall_trace
        evaluators['steps/trace_fim'] = step
        self.model.train()
        self.logger.log_scalars(evaluators, step)   