import os

import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


class TensorboardPyTorch:
    def __init__(self, config):
        self.whether_use_wandb = config.logger_config['whether_use_wandb']
        os.makedirs(config.logger_config['log_dir'])
        if self.whether_use_wandb:
            wandb.login(key=os.environ['WANDB_API_KEY'])
            wandb.init(
                entity=config.logger_config['entity'] if config.logger_config['entity'] is not None else os.environ['WANDB_ENTITY'],
                project=config.logger_config['project_name'],
                name=config.exp_name,
                config=OmegaConf.to_container(config, resolve=True),
                dir=config.logger_config['log_dir'],
                mode=config.logger_config['mode'],
                # group=config.logger_config['group'],
            )
            if len(wandb.patched["tensorboard"]) > 0:
                wandb.tensorboard.unpatch()
            wandb.tensorboard.patch(root_logdir=config.logger_config['log_dir'], pytorch=True, save=False)
            
        self.writer = SummaryWriter(log_dir=config.logger_config['log_dir'])
        # if 'layout' in config.logger_config:
        #     self.writer.add_custom_scalars(config.logger_config['layout'])


    def close(self):
        if self.whether_use_wandb:
            wandb.finish()
        self.writer.close()
        

    def flush(self):
        self.writer.flush()

    def log_graph(self, model, inp, criterion):
        if self.whether_use_wandb:
            wandb.watch(model, log_freq=1000, idx=0, log_graph=True, log='all', criterion=criterion)
        self.writer.add_graph(model, inp)
        
    def log_figures(self, images, global_step):
        for tag in images:
            self.writer.add_figure(tag, images[tag], global_step=global_step)

    def log_histogram(self, values, global_step): # problem with numpy=1.24.0
        for tag in values:
            self.writer.add_histogram(tag, values[tag], global_step=global_step)

    def log_scalars(self, scalar_dict, global_step):
        for tag in scalar_dict:
            self.writer.add_scalar(tag, scalar_dict[tag], global_step=global_step)

