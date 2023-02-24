import os
import gc
import csv
import time
import shutil
import yaml
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import adjust_learning_rate


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()

        if 'SPICE' in self.config['dataset']['data_dir']:
            from dataset.dataset_spice import SPICEWrapper
            self.dataset = SPICEWrapper(**self.config['dataset'])
            self.prefix = 'spice'
            if self.config['model']['name'] == 'SE3Transformer':
                from dataset.dataset_spice_dgl import SPICEWrapper
                self.dataset = SPICEWrapper(
                    **self.config['dataset'], cutoff=self.config['model']['cutoff'], 
                    max_num_neighbors=self.config['model']['max_num_neighbors']
                )
        elif 'ANI-1x' in self.config['dataset']['data_dir']:
            from dataset.dataset_ani1x import ANI1XWrapper
            self.dataset = ANI1XWrapper(**self.config['dataset'])
            self.prefix = 'ani1x'
            if self.config['model']['name'] == 'SE3Transformer':
                from dataset.dataset_ani1x_dgl import ANI1XWrapper
                self.dataset = ANI1XWrapper(
                    **self.config['dataset'], cutoff=self.config['model']['cutoff'], 
                    max_num_neighbors=self.config['model']['max_num_neighbors']
                )
        elif 'ANI-1' in self.config['dataset']['data_dir']:
            from dataset.dataset_ani1 import ANI1Wrapper
            self.dataset = ANI1Wrapper(**self.config['dataset'])
            self.prefix = 'ani1'
            if self.config['model']['name'] == 'SE3Transformer':
                from dataset.dataset_ani1_dgl import ANI1Wrapper
                self.dataset = ANI1Wrapper(
                    **self.config['dataset'], cutoff=self.config['model']['cutoff'], 
                    max_num_neighbors=self.config['model']['max_num_neighbors']
                )
        elif 'iso17' in self.config['dataset']['data_dir']:
            from dataset.dataset_iso17 import ISO17Wrapper
            self.dataset = ISO17Wrapper(**self.config['dataset'])
            self.prefix = 'iso17'
            if self.config['model']['name'] == 'SE3Transformer':
                from dataset.dataset_iso17_dgl import ISO17Wrapper
                self.dataset = ISO17Wrapper(
                    **self.config['dataset'], cutoff=self.config['model']['cutoff'], 
                    max_num_neighbors=self.config['model']['max_num_neighbors']
                )
        elif 'MD22' in self.config['dataset']['data_dir']:
            from dataset.dataset_md22 import MD22Wrapper
            self.dataset = MD22Wrapper(**self.config['dataset'])
            self.prefix = self.config['dataset']['data_dir'].split('/')[-1]
            if self.config['model']['name'] == 'SE3Transformer':
                from dataset.dataset_md22_dgl import MD22Wrapper
                self.dataset = MD22Wrapper(
                    **self.config['dataset'], cutoff=self.config['model']['cutoff'], 
                    max_num_neighbors=self.config['model']['max_num_neighbors']
                )
        else:
            raise NotImplementedError('Undefined dataset!')

        if self.config['model']['name'] == 'TorchMD-Net':
            self.model_prefix = 'torchmdnet'
        elif self.config['model']['name'] == 'EGNN':
            self.model_prefix = 'egnn'
        elif self.config['model']['name'] == 'SphereNet':
            self.model_prefix = 'spherenet'
        elif self.config['model']['name'] == 'SchNet':
            self.model_prefix = 'schnet'
        elif self.config['model']['name'] == 'SE3Transformer':
            self.model_prefix = 'se3transformer'
        else:
            raise NotImplementedError('Undefined model!')
        
        dir_name = '_'.join([datetime.now().strftime('%b%d_%H-%M-%S'), self.prefix, self.model_prefix])
        self.log_dir = os.path.join('runs', dir_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    @staticmethod
    def _save_config_file(ckpt_dir):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
            shutil.copy('./config.yaml', os.path.join(ckpt_dir, 'config.yaml'))

    def loss_fn(self, model, data):
        if self.config['model']['name'] == 'SE3Transformer':
            pred_e, __ = model(data, self.device)
            y = data.y.to(self.device)
            loss = F.mse_loss(
                pred_e, self.normalizer.norm(y), reduction='mean'
            )
        else:
            data = data.to(self.device)
            pred_e, __ = model(data.x, data.pos, data.batch)
            loss = F.mse_loss(
                pred_e, self.normalizer.norm(data.y), reduction='mean'
            )
        return pred_e, loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        labels = []
        for i, d in enumerate(train_loader):
            labels.append(d.y)
            if i % 5000 == 0:
                print('normalizing', i)
        labels = torch.cat(labels)
        self.normalizer = Normalizer(labels)
        print(self.normalizer.mean, self.normalizer.std, labels.shape)
        del labels
        gc.collect() # free memory

        if self.config['model']['name'] == 'TorchMD-Net':
            from models.torchmdnet import TorchMD_ET
            model = TorchMD_ET(**self.config["model"])
        elif self.config['model']['name'] == 'EGNN':
            from models.egnn import EGNN
            model = EGNN(**self.config["model"])
        elif self.config['model']['name'] == 'SchNet':
            from models.schnet import SchNetWrap
            model = SchNetWrap(**self.config["model"], auto_grad=False)
        elif self.config['model']['name'] == 'SE3Transformer':
            from models.se3transformer import SE3Transformer
            model = SE3Transformer(**self.config["model"], auto_grad=False)
        self._load_weights(model)
        model = model.to(self.device)
        
        if type(self.config['lr']) == str: self.config['lr'] = eval(self.config['lr']) 
        if type(self.config['min_lr']) == str: self.config['min_lr'] = eval(self.config['min_lr'])
        if type(self.config['weight_decay']) == str: self.config['weight_decay'] = eval(self.config['weight_decay']) 
        optimizer = AdamW(
            model.parameters(), self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

        ckpt_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        self._save_config_file(ckpt_dir)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):                
                adjust_learning_rate(optimizer, epoch_counter + bn / len(train_loader), self.config)

                __, loss = self.loss_fn(model, data)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                    print(epoch_counter, bn, 'loss', loss.item())
                    torch.cuda.empty_cache()

                n_iter += 1
            
            gc.collect() # free memory
            torch.cuda.empty_cache()

            # validate the model 
            valid_rmse = self._validate(model, valid_loader)
            self.writer.add_scalar('valid_rmse', valid_rmse, global_step=valid_n_iter)
            print('Validation', epoch_counter, 'valid rmse', valid_rmse)

            if valid_rmse < best_valid_loss:
                best_valid_loss = valid_rmse
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))

            valid_n_iter += 1
        
        start_time = time.time()
        self._test(model, test_loader)
        print('test duration:', time.time() - start_time)

    def _load_weights(self, model):
        try:
            state_dict = torch.load(os.path.join(self.config['load_model'], 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions, labels = [], []
        model.eval()

        for bn, data in enumerate(valid_loader):
            pred_e, __ = self.loss_fn(model, data)
            pred_e = self.normalizer.denorm(pred_e)

            if self.config['model']['name'] == 'SE3Transformer':
                y = data.y.to(self.device)
            else:
                y = data.y

            if self.device == 'cpu':
                predictions.extend(pred_e.flatten().detach().numpy())
                labels.extend(y.flatten().numpy())
            else:
                predictions.extend(pred_e.flatten().cpu().detach().numpy())
                labels.extend(y.cpu().flatten().numpy())
            
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        model.train()
        return mean_squared_error(labels, predictions, squared=False)
    
    def _test(self, model, test_loader):
        model_path = os.path.join(self.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded {} with success.".format(model_path))

        predictions, labels, smiles = [], [], []
        model.eval()

        for bn, data in enumerate(test_loader):                
            pred_e, __ = self.loss_fn(model, data)
            pred_e = self.normalizer.denorm(pred_e)

            if self.model_prefix == 'se3transformer':
                label = data.y.to(self.device)
            else:
                label = data.y
            smiles.extend(data.smi)

            if 'ani1' in self.prefix:
                if self.model_prefix == 'se3transformer':
                    self_energy = data.self_energy.to(self.device)
                    pred_e += self_energy
                    label += self_energy
                else:
                    pred_e += data.self_energy
                    label += data.self_energy

            if self.device == 'cpu':
                predictions.extend(pred_e.flatten().detach().numpy())
                labels.extend(label.flatten().numpy())
            else:
                predictions.extend(pred_e.flatten().cpu().detach().numpy())
                labels.extend(label.cpu().flatten().numpy())
        
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        rmse = mean_squared_error(labels, predictions, squared=False)
        mae = mean_absolute_error(labels, predictions)

        with open(os.path.join(self.log_dir, 'results.csv'), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(len(labels)):
                csv_writer.writerow([smiles[i], predictions[i], labels[i]])
            csv_writer.writerow([rmse, mae])


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()