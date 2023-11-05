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

from src.utils.prepare import prepare_model, prepare_loaders_clp, prepare_criterion, prepare_optim_and_scheduler
from src.utils.utils_trainer import manual_seed
from src.utils.utils_visualisation import ee_tensorboard_layout
from src.trainer.trainer_classification_phases import TrainerClassification
from src.modules.aux_modules import TunnelandProbing, TraceFIM
from src.modules.metrics import RunStats


def objective(exp, epochs, lr, wd, lr_lambda):
    # ════════════════════════ prepare general params ════════════════════════ #


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    model = prepare_model(type_names['model'], model_params=model_params).to(device)

    
    # ════════════════════════ prepare criterion ════════════════════════ #
    
    criterion_params = {'criterion_name': 'ce'}
    
    criterion = prepare_criterion(type_names['criterion'], criterion_params=criterion_params)    
    
    # ════════════════════════ prepare loaders ════════════════════════ #
    
    
    dataset_params = {'dataset_path': None, 'whether_aug': True, 'proper_normalization': True}
    loader_params = {'batch_size': 125, 'pin_memory': True, 'num_workers': 8}
    
    loaders = prepare_loaders_clp(type_names['dataset'], dataset_params=dataset_params, loader_params=loader_params)
    
    
    # ════════════════════════ prepare optimizer & scheduler ════════════════════════ #
    
    
    LR = lr
    WD = wd
    LR_LAMBDA = lr_lambda
    T_max = len(loaders['train']) * epochs
    optim_params = {'lr': LR, 'weight_decay': WD}
    scheduler_params = {'lr_lambda': lambda epoch: LR_LAMBDA}
    
    optim, lr_scheduler = prepare_optim_and_scheduler(model, optim_name=type_names['optim'], optim_params=optim_params, scheduler_name=type_names['scheduler'], scheduler_params=scheduler_params)
    scheduler_params['lr_lambda'] = LR_LAMBDA # problem with omegacong with primitive type
    
    # ════════════════════════ prepare wandb params ════════════════════════ #
    
    quick_name = 'lr_search'
    ENTITY_NAME = 'bartekk0'
    PROJECT_NAME = 'NeuralCollapse'
    GROUP_NAME = f'{exp}, {type_names["optim"]}, {type_names["dataset"]}, {type_names["model"]}_lr_{LR}_wd_{WD}_lr_lambda_{LR_LAMBDA}'
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
    held_out = {}
    # held_out['proper_x_left'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_left.pt').to(device)
    # held_out['proper_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_proper_x_right.pt').to(device)
    # held_out['blurred_x_right'] = torch.load(f'data/{type_names["dataset"]}_held_out_blurred_x_right.pt').to(device)
    
    
    # ════════════════════════ prepare extra modules ════════════════════════ #
    
    
    extra_modules = defaultdict(lambda: None)
    extra_modules['run_stats'] = RunStats(model, optim)
    
    # extra_modules['tunnel'] = TunnelandProbing(loaders, model, num_classes=NUM_CLASSES, optim_type=type_names['optim'], optim_params=optim_params,
    #                                            reprs_hook=extra_modules['hooks_reprs'], epochs_probing=20)
    # extra_modules['trace_fim'] = TraceFIM(held_out, model, num_classes=NUM_CLASSES)
    
    
    # ════════════════════════ prepare trainer ════════════════════════ #
    
    
    params_trainer = {
        'model': model,
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
    params_names = [n for n, p in model.named_parameters() if p.requires_grad]
    
    logger_config = {'logger_name': 'tensorboard',
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
    config.save_multi = int((T_max // epochs) * 20)
    # config.stiff_multi = (T_max // (window + epochs)) // 2
    config.tunnel_multi = int((T_max // epochs) * 10)
    config.fim_trace_multi = (T_max // epochs) // 2
    config.run_stats_multi = (T_max // epochs) // 2
    
    config.clip_value = CLIP_VALUE
    config.random_seed = RANDOM_SEED
    config.whether_disable_tqdm = True
    
    config.base_path = 'reports'
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
    wd = float(sys.argv[2])
    lr_lambda = 1.0
    EPOCHS = 300
    objective('just_run', EPOCHS, lr, wd, lr_lambda)
