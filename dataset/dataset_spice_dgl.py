import gc
import os
import h5py
import numpy as np
from rdkit import Chem
from openmm.unit import *
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import radius_graph

posScale = 1*bohr/angstrom
energyScale = 1*hartree/item/(kilojoules_per_mole)
forceScale = energyScale/posScale
ATOM_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}


class SPICE(Dataset):
    def __init__(self, cutoff, max_num_neighbors, species, positions, energies, smiles=None):
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
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
        x = self.species[index]
        y = self.energies[index]

        num_atoms = len(x)
        num_atoms = torch.tensor(num_atoms, dtype=torch.long)

        x = self.to_one_hot(np.array(x), len(ATOM_DICT))
        f = torch.tensor(x[...,None], dtype=torch.float) # [num_atoms,28,1]
        pos = torch.tensor(pos, dtype=torch.float) * posScale
        y = torch.tensor(y, dtype=torch.float).view(1,-1) * energyScale

        # edge_index = radius_graph(
        #     pos,
        #     r=self.cutoff,
        #     loop=False,
        #     max_num_neighbors=self.max_num_neighbors
        # )
        
        if self.smiles is None:
            data = Data(
                f=f, pos=pos, y=y, num_atoms=num_atoms
            )
        else:
            data = Data(
                f=f, pos=pos, y=y, 
                smi=self.smiles[index], num_atoms=num_atoms
            )
        
        return data

        # if self.smiles is None:
        #     smi = None
        # else:
        #     smi = self.smiles[index]

        # return f, pos, edge_index, y, smi

    def __len__(self):
        return len(self.positions)


def collate_fn(data):
    f, pos, edge_index, y, smi = map(list, zip(*data))
    return f, pos, edge_index, y, smi


class SPICEWrapper(object):
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

        n_mol = 0
        self.train_species, self.valid_species, self.test_species = [], [], []
        self.train_positions, self.valid_positions, self.test_positions = [], [], []
        self.train_energies, self.valid_energies, self.test_energies = [], [], []
        self.test_smiles = []

        with h5py.File(os.path.join(self.data_dir, 'SPICE.hdf5'), 'r') as raw_data:
            for name in raw_data:
                n_mol += 1
                if n_mol % 1000 == 0:
                    print('Loading # molecule %d' % n_mol)
                
                g = raw_data[name]
                smi = g['smiles'][0].decode('ascii')
                mol = Chem.MolFromSmiles(smi)
                mol = Chem.AddHs(mol)
                species = []
                for atom in mol.GetAtoms():
                    ele = atom.GetSymbol()
                    charge = atom.GetFormalCharge()
                    species.append(ATOM_DICT[(ele, charge)])
                
                n_conf = g['conformations'].shape[0]
                indices = list(range(n_conf))
                random_state.shuffle(indices)
                split1 = int(np.floor(self.valid_size * n_conf))
                split2 = int(np.floor(self.test_size * n_conf))
                valid_idx, test_idx, train_idx = \
                    indices[:split1], indices[split1:split1+split2], indices[split1+split2:]
                
                for i in train_idx:
                    self.train_positions.append(g['conformations'][i])
                    self.train_energies.append(g['formation_energy'][i])
                    self.train_species.append(species)

                for i in valid_idx:
                    self.valid_positions.append(g['conformations'][i])
                    self.valid_energies.append(g['formation_energy'][i])
                    self.valid_species.append(species)

                for i in test_idx:
                    self.test_positions.append(g['conformations'][i])
                    self.test_energies.append(g['formation_energy'][i])
                    self.test_species.append(species)
                self.test_smiles.extend([smi] * len(test_idx))
                
                # for i in range(n_conf):
                #     pos = g['conformations'][i]
                #     e = g['formation_energy'][i]
                #     ff = g['dft_total_gradient'][i]

        print("# molecules:", n_mol)
        print("# train conformations:", len(self.train_species))
        print("# valid conformations:", len(self.valid_species))
        print("# test conformations:", len(self.test_species))

        train_dataset = SPICE(
            cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            species=self.train_species, positions=self.train_positions, 
            energies=self.train_energies
        )
        valid_dataset = SPICE(
            cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            species=self.valid_species, positions=self.valid_positions, 
            energies=self.valid_energies
        )
        test_dataset = SPICE(
            cutoff=self.cutoff, max_num_neighbors=self.max_num_neighbors,
            species=self.test_species, positions=self.test_positions, 
            energies=self.test_energies, smiles=self.test_smiles
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
            shuffle=False, drop_last=False, 
        )

        del self.train_species, self.valid_species, self.test_species
        del self.train_positions, self.valid_positions, self.test_positions
        del self.train_energies, self.valid_energies, self.test_energies
        del self.test_smiles
        gc.collect()

        return train_loader, valid_loader, test_loader
