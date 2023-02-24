import os
from ase.db import connect
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


class ISO17(Dataset):
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


class ISO17Wrapper(object):
    def __init__(self, batch_size, num_workers, data_dir, **kwargs):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _read_db(self, db_path):
        species, coordinates = [], []
        energies = []
        with connect(db_path) as conn:
            for row in conn.select():
                atoms = row.toatoms()
                elements = atoms.get_atomic_numbers()
                elements = [ATOM_DICT[i] for i in elements]
                pos = atoms.get_positions()
                e = float(row['total_energy'])
                species.append(elements)
                coordinates.append(pos)
                energies.append(e)
        return species, coordinates, np.array(energies)

    def get_data_loaders(self):
        train_idx = []
        with open(os.path.join(self.data_dir, 'train_ids.txt'), 'r') as f:
            for line in f:
                train_idx.append(int(line.strip()) - 1)
        valid_idx = []
        with open(os.path.join(self.data_dir, 'validation_ids.txt'), 'r') as f:
            for line in f:
                valid_idx.append(int(line.strip()) - 1)

        species, coordinates, energies = self._read_db(os.path.join(self.data_dir, 'reference.db'))
        print(len(species), len(coordinates), len(energies), len(train_idx), len(valid_idx))
        self.train_species = [species[idx] for idx in train_idx]
        self.train_positions = [coordinates[idx] for idx in train_idx]
        self.train_energies = energies[train_idx]
        self.valid_species = [species[idx] for idx in valid_idx]
        self.valid_positions = [coordinates[idx] for idx in valid_idx]
        self.valid_energies = energies[valid_idx]

        self.test_species, self.test_positions, self.test_energies = \
            self._read_db(os.path.join(self.data_dir, 'test_other.db'))
        self.test_smiles = ['C7O2H10'] * len(self.test_energies)

        n_conf = len(self.test_species) + len(self.train_species) + len(self.valid_species)

        print("# molecules:", n_conf)
        print("# train conformations:", len(self.train_species))
        print("# valid conformations:", len(self.valid_species))
        print("# test conformations:", len(self.test_species))

        train_dataset = ISO17(
            self.data_dir, species=self.train_species,
            positions=self.train_positions, energies=self.train_energies
        )
        valid_dataset = ISO17(
            self.data_dir, species=self.valid_species,
            positions=self.valid_positions, energies=self.valid_energies
        )
        test_dataset = ISO17(
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