import os

import wandb
from omegaconf import OmegaConf


class WandbLogger:
    def __init__(self, config):
        self.project = config.logger_config['project_name']
        self.writer = wandb
        self.writer.login(key=os.environ['WANDB_API_KEY'])
        if not os.path.isdir(config.logger_config['log_dir']):
            os.makedirs(config.logger_config['log_dir'])
        self.writer.init(
            entity=config.logger_config['entity'] if config.logger_config['entity'] is not None else os.environ['WANDB_ENTITY'],
            project=config.logger_config['project_name'],
            name=config.exp_name,
            config=OmegaConf.to_container(config, resolve=True),
            dir=config.logger_config['log_dir'],
            mode=config.logger_config['mode'],
            # group=config.logger_config['group'],
        )

    def close(self):
        self.writer.finish()

    def log_model(self, model, criterion, log, log_freq: int=1000, log_graph: bool=True):
        self.writer.watch(model, criterion, log=log, log_freq=log_freq, log_graph=log_graph)

    def log_histogram(self, tag, tensor, global_step): # problem with numpy=1.24.0
        tensor = tensor.view(-1, 1)
        self.writer.log({tag: wandb.Histogram(tensor)}, step=global_step)

    def log_scalars(self, evaluators, global_step=None):
        self.writer.log(evaluators)
        
    def log_plots(self, plot_images):
        self.writer.log(plot_images)


