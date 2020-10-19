import os

import torch


class Saver:
    def __init__(self, configs):
        super(Saver, self).__init__()
        self.configs = configs
        self.path = configs.save_path

    def save(self, model):
        filename = self.configs.model + ".mdl"
        content = {
            "configs": self.configs,
            "state_dict": model.state_dict()
        }
        torch.save(content, os.path.join("../", self.path, filename))