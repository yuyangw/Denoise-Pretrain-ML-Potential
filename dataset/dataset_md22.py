import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from rdkit import Chem

ATOM_DICT = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
PAIR_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}


class MD22(Dataset):
    def __init__(self, data_dir, species, positions, energies, smiles=None):
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.smiles = smiles

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]

        x = []
        for atom in atoms:
            x.append(PAIR_DICT[(atom, 0)])
        x = torch.tensor(x, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).view(1,-1)
        
        if self.smiles is None:
            data = Data(x=x, pos=pos, y=y)
        else:
            data = Data(x=x, pos=pos, y=y, smi=self.smiles[index])
        
        return data

    def __len__(self):
        return len(self.positions)


class MD22Wrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, test_size, data_dir, seed):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed
    
    def _read_xyz(self, xyz_path):
        species, coordinates, energies = [], [], []
        elements, coords = [], []

        with open(xyz_path, 'r') as f:
            for i, line in enumerate(f):
                l_list = line.strip().split()
                if len(l_list) == 2:
                    if len(elements) > 0:
                        species.append(elements)
                        coordinates.append(coords)
                    elements = []
                    coords = []

                    e = l_list[0].replace('Energy=', '')
                    energies.append(float(e))

                elif len(l_list) == 7:
                    ele, x, y, z = l_list[:4]
                    point = [float(x), float(y), float(z)]
                    elements.append(ele)
                    coords.append(point)

        species.append(elements)
        coordinates.append(coords)

        return species, coordinates, np.array(energies)

    def get_data_loaders(self):
        random_state = np.random.RandomState(seed=self.seed)

        # self.species, self.positions, self.energies = [], [], []
        self.train_species, self.valid_species, self.test_species = [], [], []
        self.train_positions, self.valid_positions, self.test_positions = [], [], []
        self.train_energies, self.valid_energies, self.test_energies = [], [], []
        self.test_smiles = []
        
        species, coordinates, energies = self._read_xyz(self.data_dir)

        assert len(species) == len(coordinates) == len(energies)
        n_conf = len(species)
        indices = list(range(n_conf))
        random_state.shuffle(indices)
        split1 = int(np.floor(self.valid_size * n_conf))
        split2 = int(np.floor(self.test_size * n_conf))
        valid_idx, test_idx, train_idx = \
            indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

        self.train_species.extend([species[idx] for idx in train_idx])
        self.train_energies = energies[train_idx]
        for i in train_idx:
            self.train_positions.append(coordinates[i])
        
        self.valid_species.extend([species[idx] for idx in valid_idx])
        self.valid_energies = energies[valid_idx]
        for i in valid_idx:
            self.valid_positions.append(coordinates[i])
        
        self.test_species.extend([species[idx] for idx in test_idx])
        self.test_energies = energies[test_idx]
        for i in test_idx:
            self.test_positions.append(coordinates[i])
        self.test_smiles.extend([self.data_dir.split('/')[-1]] * len(test_idx))

        print("# conformations:", len(species))
        print("# train conformations:", len(self.train_species))
        print("# valid conformations:", len(self.valid_species))
        print("# test conformations:", len(self.test_species))

        train_dataset = MD22(
            self.data_dir, species=self.train_species,
            positions=self.train_positions, energies=self.train_energies
        )
        valid_dataset = MD22(
            self.data_dir, species=self.valid_species,
            positions=self.valid_positions, energies=self.valid_energies
        )
        test_dataset = MD22(
            self.data_dir, species=self.test_species, 
            positions=self.test_positions, energies=self.test_energies, 
            smiles=self.test_smiles
        )

        train_loader = PyGDataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=True, drop_last=True, 
            pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        valid_loader = PyGDataLoader(
            valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        test_loader = PyGDataLoader(
            test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=False,
            pin_memory=True, persistent_workers=(self.num_workers > 0)
        )

        del self.train_species, self.valid_species, self.test_species
        del self.train_positions, self.valid_positions, self.test_positions
        del self.train_energies, self.valid_energies, self.test_energies
        del self.test_smiles

        return train_loader, valid_loader, test_loader