import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

from paint3d import utils


def init_dataloaders(cfg, device=torch.device("cpu")):
    init_train_dataloader = MultiviewDataset(cfg.render, device=device).dataloader()
    val_large_loader = ViewsDataset(cfg.render, device=device, size=cfg.log.full_eval_size).dataloader()
    dataloaders = {'train': init_train_dataloader, 'val_large': val_large_loader}
    return dataloaders


class MultiviewDataset:
    def __init__(self, cfg, device):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, tests
        size = self.cfg.n_views
        self.phis = [(index / size) * 360 for index in range(size)]
        self.thetas = [self.cfg.base_theta for _ in range(size)]

        # Alternate lists
        alternate_lists = lambda l: [l[0]] + [i for j in zip(l[1:size // 2], l[-1:size // 2:-1]) for i in j] + [
            l[size // 2]]
        if self.cfg.alternate_views: 
            self.phis = alternate_lists(self.phis)
            self.thetas = alternate_lists(self.thetas)

        for phi, theta in self.cfg.views_before:
            self.phis = [phi] + self.phis
            self.thetas = [theta] + self.thetas
        for phi, theta in self.cfg.views_after:
            self.phis = self.phis + [phi]
            self.thetas = self.thetas + [theta]

        self.size = len(self.phis)

    def collate(self, index):
        phi = self.phis[index[0]]
        theta = self.thetas[index[0]]
        radius = self.cfg.radius
        thetas = torch.FloatTensor([np.deg2rad(theta)]).to(self.device).item()
        phis = torch.FloatTensor([np.deg2rad(phi)]).to(self.device).item()

        return {'theta': thetas, 'phi': phis, 'radius': radius}

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  
        return loader

class ViewsDataset:
    def __init__(self, cfg, device, size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test
        self.size = size

    def collate(self, index):
        phi = (index[0] / self.size) * 360
        thetas = torch.FloatTensor([np.deg2rad(self.cfg.base_theta)]).to(self.device).item()
        phis = torch.FloatTensor([np.deg2rad(phi)]).to(self.device).item()
        return {'theta': thetas,  'phi': phis, 'radius': self.cfg.radius}

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=False,
                            num_workers=0)
        loader._data = self  
        return loader
