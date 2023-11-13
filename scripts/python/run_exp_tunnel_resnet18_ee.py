#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from math import ceil

import omegaconf

# from rich.traceback import install
# install(show_locals=True)

from src.utils.prepare import prepare_model, prepare_loaders, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification_ee import TrainerClassification

from src.modules.hooks import Hooks
from src.modules.aux_modules import TunnelandProbing, TraceFIM
from src.modules.aux_modules_collapse import TunnelGrad
from src.modules.metrics import RunStats
from src.modules.tunnel_code import RepresentationsSpectra
from src.modules.early_exit import SDN


def objective(exp, epochs, lr, wd, momentum, lr_lambda, checkpoint_path, phase1):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs -= phase1
    NUM_CLASSES = 10
    RANDOM_SEED = 83
    
    type_names = {
        'model': 'resnet_tunnel',
        'criterion': 'cls',
        'dataset': 'cifar10',
        'optim': 'sgd',
        'scheduler': 'multiplicative'
    }
    
    
    # ════════════════════════ prepare seed ════════════════════════ #
    
    
    manual_seed(random_seed=RANDOM_SEED, device=device)
    
    
    # ════════════════════════ prepare model ════════════════════════ #
    
    
    model_config = {'backbone_type': 'resnet18',
                    'only_features': False,
                    'batchnorm_layers': True,
                    'width_scale': 1.0,
                    'skips': True,
                    'modify_resnet': True}
    model_params = {'model_config': model_config, 'num_classes': NUM_CLASSES, 'dataset_name': type_names['dataset']}
    
    model = prepare_model(type_names['model'], model_params=model_params, checkpoint_path=checkpoint_path).to(device)

    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    criterion_params = {'criterion_name': 'ce'}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = lr
    MOMENTUM = momentum
    WD = wd
    LR_LAMBDA = lr_lambda
    T_max = len(loaders['train']) * epochs
    optim_params = {'lr': LR, 'weight_decay': WD, 'momentum': MOMENTUM}
    scheduler_params = {'lr_lambda': lambda epoch: LR_LAMBDA}
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    scheduler_params['lr_lambda'] = LR_LAMBDA # problem with omegacong with primitive type
    
    # ════════════════════════ prepare wandb params ════════════════════════ #
    
    quick_name = f'early_exit_sdn_phase1_{phase1}'
    ENTITY_NAME = 'gmum'
    PROJECT_NAME = 'NeuralCollapse'
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_lr_{LR}_momentum_{MOMENTUM}_wd_{WD}_lr_lambda_{LR_LAMBDA}'
    EXP_NAME = f'{GROUP_NAME}, {quick_name}'

    h_params_overall = {
        'model': model_params,
        'criterion': criterion_params,
        'dataset': dataset_params,
        'loaders': loader_params,
        'optim': optim_params,
        'scheduler': scheduler_params,
        'type_names': type_names
    }   
 
 
    # ════════════════════════ prepare held out data ════════════════════════ #
    
    
    # DODAJ - POPRAWNE DANE
    print('liczba parametrów', sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values()))
    print("device", device)
    held_out = {}
    data = '/net/pr2/projects/plgrid/plgg_ccbench/data'
    # held_out['x_held_out'] = torch.load(f'{data}/{type_names["dataset"]}_held_out_proper_x.pt').to(device)
    # held_out['y_held_out'] = torch.load(f'{data}/{type_names["dataset"]}_held_out_proper_y.pt').to(device)
    
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    model_ee = SDN(model, criterion=criterion, ic_idxs=[0,1,2,3,4,5,6], confidence_threshold=0.8, is_model_frozen=False, prob_conf=True, sample=next(iter(loaders['train']))[0])
    extra_modules = defaultdict(lambda: None)
    # extra_modules['hooks_reprs'] = Hooks(model, logger=None, callback_type='gather_reprs', kwargs_callback={"cutoff": 4000})
    # extra_modules['hooks_reprs'].register_hooks([torch.nn.Conv2d, torch.nn.Linear])
    
    # extra_modules['run_stats'] = RunStats(model, optim)
    
    extra_modules['tunnel_code'] = RepresentationsSpectra(model, loader=loaders['test'], modules_list=[torch.nn.Conv2d, torch.nn.Linear], MAX_REPR_SIZE=8000)
    
    # extra_modules['tunnel'] = TunnelandProbing(loaders, model, num_classes=NUM_CLASSES, optim_type=type_names['optim'], optim_params={'lr': 1e-2, 'weight_decay': 0.0},
    #                                            reprs_hook=extra_modules['hooks_reprs'], epochs_probing=50)
    
    # extra_modules['tunnel_grads'] = TunnelGrad(loaders, model, cutoff=4000)
    # extra_modules['trace_fim'] = TraceFIM(held_out['x_held_out'], model, num_classes=NUM_CLASSES)
    
    
    # ════════════════════════ prepare trainer ════════════════════════ #
    
    
    params_trainer = {
        'model': model_ee,
        'criterion': criterion,
        'loaders': loaders,
        'optim': optim,
        'lr_scheduler': lr_scheduler,
        'device': device,
        'extra_modules': extra_modules,
    }
    
    trainer = TrainerClassification(**params_trainer)


    # ════════════════════════ prepare run ════════════════════════ #


    CLIP_VALUE = 0.0
    params_names = [n for n, p in model_ee.named_parameters() if p.requires_grad]
    
    logger_config = {'logger_name': 'wandb',
                     'project_name': PROJECT_NAME,
                     'entity': ENTITY_NAME,
                     'hyperparameters': h_params_overall,
                     'whether_use_wandb': True,
                     'layout': ee_tensorboard_layout(params_names), # is it necessary?
                     'mode': 'online',
    }
    
    config = omegaconf.OmegaConf.create()
    
    config.epoch_start_at = 0
    config.epoch_end_at = epochs
    
    config.log_multi = 1#(T_max // epochs) // 10
    config.save_multi = 0#int((T_max // epochs) * 40)
    config.run_stats_multi = (T_max // epochs) // 2
    config.tunnel_multi = int((T_max // epochs) * 10)
    config.tunnel_grads_multi = int((T_max // epochs) * 10)
    config.fim_trace_multi = (T_max // epochs) // 2
    # config.stiff_multi = (T_max // (window + epochs)) // 2
    
    config.clip_value = CLIP_VALUE
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = '/net/pr2/projects/plgrid/plgg_ccbench/bartek/reports2'
    config.exp_name = EXP_NAME
    config.logger_config = logger_config
    config.checkpoint_path = None
    
    
    # ════════════════════════ run ════════════════════════ #
    
    
    if exp == 'just_run':
        trainer.run_exp(config)
    elif exp == 'blurring':
        trainer.run_exp_blurred(config)
    elif exp == 'half':
        trainer.run_exp_blurred(config)
    else:
        raise ValueError('wrong exp name have been chosen')


if __name__ == "__main__":
    lr = float(sys.argv[1])
    momentum = float(sys.argv[2])
    is_wd = int(sys.argv[3])
    checkpoint_path = str(sys.argv[4])
    phase1 = int(sys.argv[5])
    wd = 1e-4 * 1e-1 / lr * is_wd
    lr_lambda = 1.0
    EPOCHS = 200
    objective('just_run', EPOCHS, lr, wd, momentum, lr_lambda, checkpoint_path, phase1)
