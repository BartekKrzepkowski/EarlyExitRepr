from src.modules.callbacks import CALLBACK_TYPE

class Hooks:
    def __init__(self, model, logger, callback_type, kwargs_callback={}):
        self.model = model
        self.logger = logger
        self.callback = CALLBACK_TYPE[callback_type](**kwargs_callback)
        self.hooks = []
        
    def register_hooks(self, modules_list):
        self.hooks = []
        for module in self.model.modules():
            if any(isinstance(module, module_type) for module_type in modules_list):
                self.hooks.append(module.register_forward_hook(self.callback))
                
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def write_to_tensorboard(self, step):
        self.callback.prepare()
        self.logger.log_scalars(self.callback.get_assets(), step)
        self.reset()
            
    def reset(self):
        self.callback.reset()
        
    def disable(self):
        self.callback.disable()
        
    def enable(self):
        self.callback.enable()
        
    def get_assets(self):
        return self.callback.get_assets()