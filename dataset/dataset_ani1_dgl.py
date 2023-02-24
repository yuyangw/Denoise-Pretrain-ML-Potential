import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import radius_graph

from dataset.utils import anidataloader

ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}
SELF_INTER_ENERGY = {
    'H': -0.500607632585, 
    'C': -37.8302333826,
    'N': -54.5680045287,
    'O': -75.0362229210
}

class ANI1(Dataset):
    def __init__(self, cutoff, max_num_neighbors, data_dir, species, positions, energies, smiles=None):
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.data_dir = data_dir
        self.species = species
        self.positions = positions
        self.energies = energies
        self.smiles = smiles
    
    def to_one_hot(self, data, num_classes):
        one_hot = np.zeros(list(data.shape) + [num_classes])
        one_hot[np.arange(len(data)),data] = 1
        return one_hot

    def __getitem__(self, index):
        pos = self.positions[index]
        atoms = self.species[index]
        y = self.energies[index]

        num_atoms = len(atoms)
        num_atoms = torch.tensor(num_atoms, dtype=torch.long)

        x = []
        self_energy = 0.0
        for atom in atoms:
            # x.append(ATOM_DICT[atom])
            x.append(ATOM_DICT[(atom, 0)])
            self_energy += SELF_INTER_ENERGY[atom]
        x = self.to_one_hot(np.array(x), len(ATOM_DICT))
        f = torch.tensor(x[...,None], dtype=torch.float) # [num_atoms,28,1]
        pos = torch.tensor(pos, dtype=torch.float)

        # Hartree to kcal/mol
        y = torch.tensor(y, dtype=torch.float).view(1,-1) * 627.5
        # Hartree to kcal/mol
        self_energy = torch.tensor(self_energy, dtype=torch.float).view(1,-1) * 627.5

        if self.smiles is None:
            data = Data(
                f=f, pos=pos, y=y-self_energy, 
                self_energy=self_energy, num_atoms=num_atoms
            )
        else:
            data = Data(
                f=f, pos=pos, y=y-self_energy, self_energy=self_energy, 
                smi=self.smiles[index], num_atoms=num_atoms
            )
        
        return data

    def __len__(self):
        return len(self.positions)


class ANI1Wrapper(object):
    def __init__(self, 
        batch_size, num_workers, valid_size, test_size, data_dir, 
        seed, cutoff, max_num_neighbors, **kwargs
    ):
        super(object, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.seed = seed

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def get_data_loaders(self):
        random_state = np.random.RandomState(seed=self.seed)

        # read the data
        hdf5files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]

        curr_idx, n_mol = 0, 0
        self.train_species, self.valid_species, self.test_species = [], [], []
        self.train_positions, self.valid_positions, self.test_positions = [], [], []
        self.train_energies, self.valid_energies, self.test_energies = [], [], []
        self.test_smiles = []
        
        for f in hdf5files:
            print('reading:', f)
            h5_loader = anidataloader(os.path.join(self.data_dir, f))
            for data in h5_loader:
                X = data['coordinates']
                S = data['species']
                E = data['energies']
                smi = ''.join(data['smiles'])
                
                n_conf = E.shape[0]
                indices = list(range(n_conf))
                random_state.shuffle(indices)
                split1 = int(np.floor(self.valid_size * n_conf))
                split2 = int(np.floor(self.test_size * n_conf))
                valid_idx, test_idx, train_idx = \
                    indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

                self.train_species.extend([S] * len(train_idx))
                self.train_energies.append(E[train_idx])
                for i in train_idx:
                    self.train_positions.append(X[i])
                
                self.valid_species.extend([S] * len(valid_idx))
                self.valid_energies.append(E[valid_idx])
                for i in valid_idx:
                    self.valid_positions.append(X[i])
                
                self.test_species.extend([S] * len(test_idx))
                self.test_energies.append(E[test_idx])
                for i in test_idx:
                    self.test_positions.append(X[i])
                self.test_smiles.extend([smi] * len(test_idx))

                n_mol += 1
            
            h5_loader.cleanup()
        
        self.train_energies = np.concatenate(self.train_energies, axis=0)
        self.valid_energies = np.concatenate(self.valid_energies, axis=0)
        self.test_energies = np.concatenate(self.test_energies, axis=0)

        print("# molecules:", n_mol)
        print("# train conformations:", len(self.train_species))
        print("# valid conformations:", len(self.valid_species))
        print("# test conformations:", len(self.test_species))

        train_dataset = ANI1(
            cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            data_dir=self.data_dir, species=self.train_species,
            positions=self.train_positions, energies=self.train_energies
        )
        valid_dataset = ANI1(
            cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            data_dir=self.data_dir, species=self.valid_species,
            positions=self.valid_positions, energies=self.valid_energies
        )
        test_dataset = ANI1(
            cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            data_dir=self.data_dir, species=self.test_species, 
            positions=self.test_positions, energies=self.test_energies, 
            smiles=self.test_smiles
        )

        train_loader = PyGDataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=True, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
        valid_loader = PyGDataLoader(
            valid_dataset, batch_size=self.batch_size//8, num_workers=self.num_workers, 
            shuffle=False, drop_last=True, 
            pin_memory=True, persistent_workers=True
        )
        test_loader = PyGDataLoader(
            test_dataset, batch_size=self.batch_size//8, num_workers=self.num_workers, 
            shuffle=False, drop_last=False
        )

        del self.train_species, self.valid_species, self.test_species
        del self.train_positions, self.valid_positions, self.test_positions
        del self.train_energies, self.valid_energies, self.test_energies
        del self.test_smiles

        return train_loader, valid_loader, test_loader
