import os
import gc
import csv
import shutil
import yaml
import numpy as np
from datetime import datetime

import torch
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import adjust_learning_rate


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()

        if self.config['model']['name'] == 'SE3Transformer':
            from dataset.dataset_pretrain_dgl import PretrainDataWrapper
            self.dataset = PretrainDataWrapper(**self.config['dataset'])
        else:
            from dataset.dataset_pretrain import PretrainDataWrapper
            self.dataset = PretrainDataWrapper(**self.config['dataset'])
        
        self.prefix = ''
        if self.config['dataset']['ani1'] == True:
            self.prefix += '_ani1'
        if self.config['dataset']['ani1x'] == True:
            self.prefix += '_ani1x'
        
        if self.config['model']['name'] == 'TorchMD-Net':
            self.model_prefix = '_torchmdnet'
        elif self.config['model']['name'] == 'EGNN':
            self.model_prefix = '_egnn'
        elif self.config['model']['name'] == 'SchNet':
            self.model_prefix = '_schnet'
        elif self.config['model']['name'] == 'SE3Transformer':
            self.model_prefix = '_se3transformer'
        else:
            raise NotImplementedError('Undefined model!')

        dir_name = datetime.now().strftime('%b%d_%H-%M-%S') + self.prefix + self.model_prefix
        self.log_dir = os.path.join('runs_pretrain', dir_name)
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
            shutil.copy('./config_pretrain.yaml', os.path.join(ckpt_dir, 'config_pretrain.yaml'))

    def loss_fn(self, model, data):
        if self.config['model']['name'] == 'SE3Transformer':
            __, pred_noise = model(data, self.device)
            loss = F.mse_loss(
                pred_noise, data.noise.to(self.device), reduction='sum'
            )
        else:
            data = data.to(self.device)
            __, pred_noise = model(data.x, data.pos, data.batch)
            loss = F.mse_loss(
                pred_noise, data.noise, reduction='sum'
            )

        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        if self.config['model']['name'] == 'TorchMD-Net':
            from models.torchmdnet import TorchMD_ET
            model = TorchMD_ET(**self.config["model"])
        elif self.config['model']['name'] == 'EGNN':
            from models.egnn import EGNN
            model = EGNN(**self.config["model"])
        elif self.config['model']['name'] == 'SchNet':
            from models.schnet import SchNetWrap
            model = SchNetWrap(**self.config["model"], auto_grad=True)
        elif self.config['model']['name'] == 'SE3Transformer':
            from models.se3transformer import SE3Transformer
            model = SE3Transformer(**self.config["model"], auto_grad=True)
        self._load_weights(model)
        model = model.to(self.device)
        model.train()

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

                loss = self.loss_fn(model, data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=n_iter)
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))
                    print(epoch_counter, bn, 'loss', loss.item())
                    torch.cuda.empty_cache()
                    gc.collect() # free memory

                n_iter += 1

            gc.collect() # free memory
            torch.cuda.empty_cache()

            # validate the model 
            valid_loss = self._validate(model, valid_loader)
            self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
            print('Validation', epoch_counter, 'valid loss', valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'model.pth'))

            valid_n_iter += 1

    def _load_weights(self, model):
        try:
            checkpoints_folder = os.path.join('runs_denoise', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        valid_loss = 0
        model.eval()

        for bn, data in enumerate(valid_loader):                
            loss = self.loss_fn(model, data)
            valid_loss += loss.item()
            torch.cuda.empty_cache()
        
        gc.collect() # free memory

        model.train()
        return valid_loss / (bn+1)


if __name__ == "__main__":
    config = yaml.load(open("config_pretrain.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    trainer = Trainer(config)
    trainer.train()