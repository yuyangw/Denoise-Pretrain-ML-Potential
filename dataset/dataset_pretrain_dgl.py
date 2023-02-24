import os
import h5py
import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from openmm.unit import *

import torch
from torch.utils.data import Dataset
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from dataset.utils import ani1x_iter_data_buckets, anidataloader

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


ATOM_DICT = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
PAIR_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
posScale = 1*bohr/angstrom


class PretrainData(Dataset):
    def __init__(self, species, positions, smiles, std):
        self.species = species
        self.positions = positions
        self.smiles = smiles
        self.std = std

    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)),data] = 1
        return one_hot

    def __getitem__(self, index):
        ori_pos = self.positions[index]
        atoms = self.species[index]

        num_atoms = len(atoms)
        num_atoms = torch.tensor(num_atoms, dtype=torch.long)

        atoms = self.to_one_hot(np.array(atoms), len(PAIR_DICT))

        # add noise
        noise = np.random.normal(0, self.std, ori_pos.shape)
        pos = ori_pos + noise
        
        f = torch.tensor(atoms[...,None], dtype=torch.float) # [num_atoms,28,1]
        noise = torch.tensor(noise, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)

        data = Data(f=f, pos=pos, noise=noise, num_atoms=num_atoms)
        return data

    def __len__(self):
        return len(self.positions)


class PretrainDataWrapper(object):
    def __init__(self, 
        batch_size, num_workers, valid_size, ani1, ani1x, spice, 
        std, seed, **kwargs
    ):
        super(object, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.ani1 = ani1
        self.ani1x = ani1x
        self.spice = spice
        self.std = std
        self.seed = seed

    def get_data_loaders(self):
        random_state = np.random.RandomState(seed=self.seed)

        n_mol = 0
        self.train_species, self.valid_species = [], []
        self.train_positions, self.valid_positions = [], []
        self.train_smiles, self.valid_smiles = [], []
        
        # read ANI-1 data
        if self.ani1:
            print('Loading ANI-1 data...')
            hdf5files = [f for f in os.listdir('../ANI-1_release') if f.endswith('.h5')]
            for f in hdf5files:
                print('reading:', f)
                h5_loader = anidataloader(os.path.join('../ANI-1_release', f))
                for data in h5_loader:
                    n_mol += 1
                    if n_mol % 1000 == 0:
                        print('Loading # molecule %d' % n_mol)
                    
                    X = data['coordinates']
                    S = data['species']
                    E = data['energies']
                    
                    n_conf = E.shape[0]
                    indices = list(range(n_conf))
                    random_state.shuffle(indices)
                    split = int(np.floor(self.valid_size * n_conf))
                    valid_idx, train_idx = indices[:split], indices[split:]

                    species = [PAIR_DICT[(ele, 0)] for ele in S]
                    self.train_species.extend([species] * len(train_idx))
                    self.train_smiles.extend([''] * len(train_idx))
                    for i in train_idx:
                        self.train_positions.append(X[i])

                    species = [PAIR_DICT[(ele, 0)] for ele in S]
                    self.valid_species.extend([species] * len(valid_idx))
                    self.valid_smiles.extend([''] * len(valid_idx))
                    for i in valid_idx:
                        self.valid_positions.append(X[i])
                
                h5_loader.cleanup()

        # read ANI-1x data
        if self.ani1x:
            print('Loading ANI-1x data...')
            data_path = '../ANI-1x/ani1x_release.h5'
            data_keys = ['wb97x_dz.energy','wb97x_dz.forces'] # Original ANI-1x data (https://doi.org/10.1063/1.5023802)

            # extracting DFT/DZ energies and forces
            for data in ani1x_iter_data_buckets(data_path, keys=data_keys):
                n_mol += 1
                if n_mol % 1000 == 0:
                    print('Loading # molecule %d' % n_mol)
                
                X = data['coordinates']
                S = data['atomic_numbers']
                E = data['wb97x_dz.energy']
                S = [PAIR_DICT[(ATOM_DICT[c], 0)] for c in S]

                n_conf = E.shape[0]
                indices = list(range(n_conf))
                random_state.shuffle(indices)
                split = int(np.floor(self.valid_size * n_conf))
                valid_idx, train_idx = indices[:split], indices[split:]

                self.train_species.extend([S] * len(train_idx))
                self.train_smiles.extend([''] * len(train_idx))
                for i in train_idx:
                    self.train_positions.append(X[i])
                
                self.valid_species.extend([S] * len(valid_idx))
                self.valid_smiles.extend([''] * len(valid_idx))
                for i in valid_idx:
                    self.valid_positions.append(X[i])

        print("# molecules:", n_mol)
        print("# train conformations:", len(self.train_species))
        print("# valid conformations:", len(self.valid_species))

        train_dataset = PretrainData(
            species=self.train_species, positions=self.train_positions, 
            smiles=self.train_smiles, std=self.std,
        )
        valid_dataset = PretrainData(
            species=self.valid_species, positions=self.valid_positions, 
            smiles=self.valid_smiles, std=self.std,
        )

        train_loader = PyGDataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, drop_last=True, pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        valid_loader = PyGDataLoader(
            valid_dataset, batch_size=self.batch_size//8, num_workers=self.num_workers,
            shuffle=False, drop_last=False, pin_memory=True, persistent_workers=(self.num_workers > 0)
        )

        del self.train_species, self.valid_species
        del self.train_positions, self.valid_positions
        del self.train_smiles, self.valid_smiles

        return train_loader, valid_loader
